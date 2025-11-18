"""
Clean, self-contained Streamlit app (appv3.py) — Option A rebuild
Features included:
- Upload AOI (KML/KMZ) and Overlay (KML/KMZ)
- Grid generation (existing simplified approach expects list of shapely polygons `cells_ll`)
- Edit AOI on a map preview page (leafmap) and re-upload edited AOI
- Compute overlay vs grid intersection areas (accurate UTM areas)
- PDF report generator with extra page: Grid Area Inside Overlay Boundary
- Invasive Grid Fall map generation (Esri basemap, clipped grids labelled)
- Animated download buttons for Grid KML, Labeled KML, PDF, Grid Shapefiles
- Shapefile export (Grid Only & Grid with ID) as .zip

Notes:
- This file assumes common geospatial libs are installed: geopandas, shapely, fiona, pyproj, matplotlib, contextily, leafmap, folium, fpdf, qrcode, pillow
- Basemap downloads (contextily) can fail in some environments; code has fallbacks.
- Tweak styling values as needed.
"""

import os
import tempfile
import zipfile
import math
import base64
from io import BytesIO
from pathlib import Path

import streamlit as st
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
import qrcode
from shapely.ops import unary_union
from shapely.geometry import shape, mapping
from fpdf import FPDF
from pyproj import CRS
from PIL import Image

# Optional interactive map for editing (leafmap wrapper). If unavailable, editing UI falls back to file re-upload
try:
    import leafmap.foliumap as leafmap
    LEAFMAP_AVAILABLE = True
except Exception:
    LEAFMAP_AVAILABLE = False

st.set_page_config(page_title="TN Forest KML→Grid v3", layout="wide")

# -------------------------------
# Helpers
# -------------------------------

def utm_crs_for_lonlat(lon, lat):
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def export_shapefile_zip(gdf, name="layer"):
    tmpdir = tempfile.mkdtemp()
    shp_prefix = os.path.join(tmpdir, name)
    gdf.to_file(shp_prefix + ".shp")
    zip_path = os.path.join(tmpdir, f"{name}.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        for ext in ["shp", "shx", "dbf", "prj", "cpg"]:
            p = f"{shp_prefix}.{ext}"
            if os.path.exists(p):
                z.write(p, arcname=os.path.basename(p))
    with open(zip_path, "rb") as f:
        return f.read()


def styled_download_button(label, data_bytes, file_name, mime, icon="⬇️", bg="#0284c7"):
    try:
        b64 = base64.b64encode(data_bytes).decode()
    except Exception:
        b64 = base64.b64encode(str(data_bytes).encode("utf-8")).decode()
    html = f"""
    <style>
    .dlbtn {{ display:inline-block; position:relative; padding:10px 16px; border-radius:10px; color:#fff; font-weight:700;
              text-decoration:none; transition: transform 0.18s cubic-bezier(.2,.8,.2,1), box-shadow 0.18s ease;
              box-shadow: 0 6px 18px rgba(0,0,0,0.12); margin:6px 6px; }}
    .dlbtn:hover {{ transform: translateY(-4px) scale(1.03); box-shadow: 0 12px 30px rgba(0,0,0,0.2); }}
    .dlbtn .shine {{ position:absolute; top:0; left:-75%; width:50%; height:100%; background:linear-gradient(90deg, rgba(255,255,255,0.06), rgba(255,255,255,0.14)); transform:skewX(-20deg); transition:left 0.9s ease; }}
    .dlbtn:hover .shine {{ left:150%; }}
    </style>
    <a download="{file_name}" href="data:{mime};base64,{b64}" target="_blank" style="text-decoration:none">
      <div class="dlbtn" style="background:{bg};">
        <span style="display:inline-block;padding-right:8px">{icon}</span>
        <span style="vertical-align:middle">{label}</span>
        <div class="shine"></div>
      </div>
    </a>
    """
    st.markdown(html, unsafe_allow_html=True)


def compute_overlay_area_by_grid(cells_ll, overlay_gdf):
    if overlay_gdf is None or overlay_gdf.empty:
        return pd.DataFrame([], columns=["grid_id", "intersection_area_ha"]), 0.0, 0.0
    overlay_union = unary_union(overlay_gdf.geometry)
    centroid = overlay_union.centroid
    utm_overlay = utm_crs_for_lonlat(centroid.x, centroid.y)
    overlay_area_ha = (gpd.GeoSeries([overlay_union], crs="EPSG:4326").to_crs(utm_overlay).area.iloc[0] / 10000.0)
    rows = []
    for i, cell in enumerate(cells_ll, start=1):
        inter = cell.intersection(overlay_union)
        if inter.is_empty:
            continue
        c_centroid = cell.centroid
        utm = utm_crs_for_lonlat(c_centroid.x, c_centroid.y)
        inter_area_ha = (gpd.GeoSeries([inter], crs="EPSG:4326").to_crs(utm).area.iloc[0] / 10000.0)
        rows.append({"grid_id": int(i), "intersection_area_ha": round(inter_area_ha, 4)})
    df = pd.DataFrame(rows)
    total_grid_area_ha = df["intersection_area_ha"].sum() if not df.empty else 0.0
    return df, overlay_area_ha, total_grid_area_ha


# -------------------------------
# PDF Builder (simplified, robust)
# -------------------------------
class PDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(80, 80, 80)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")


def build_pdf_report_standard(cells_ll, merged_ll, user_inputs, cell_size, overlay_gdf, title_text, density, area_invasive, labeled_kml=None):
    MAP_X, MAP_Y, MAP_W, MAP_H, LEGEND_GAP = 15, 55, 180, 145, 8
    EMBLEM_PATH = os.path.join(os.path.dirname(__file__), "tn_emblem.png")

    pdf = PDF("P", "mm", "A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    # Page 1
    pdf.add_page()
    if os.path.exists(EMBLEM_PATH):
        try:
            pdf.image(EMBLEM_PATH, x=93, y=8, w=25)
        except Exception:
            pass
    pdf.set_y(35)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "FOREST DEPARTMENT", ln=1, align="C")
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, title_text or "", ln=1, align="C")

    # map generation
    tmp_dir = tempfile.gettempdir()
    map_img = os.path.join(tmp_dir, "map_overlay.png")
    fig, ax = plt.subplots(figsize=(7, 5.8))
    try:
        merged_gdf = gpd.GeoSeries([merged_ll], crs="EPSG:4326").to_crs(3857)
        grid_gdf = gpd.GeoSeries(cells_ll, crs="EPSG:4326").to_crs(3857)
        merged_gdf.boundary.plot(ax=ax, color="red", linewidth=3)
        grid_gdf.boundary.plot(ax=ax, color="red", linewidth=1)
        if overlay_gdf is not None and not overlay_gdf.empty:
            overlay_gdf.to_crs(3857).boundary.plot(ax=ax, color="#FFD700", linewidth=3)
        try:
            ctx.add_basemap(ax, crs=3857, source=ctx.providers.Esri.WorldImagery, attribution=False)
        except Exception:
            pass
        ax.axis("off")
        plt.tight_layout(pad=0.1)
        fig.savefig(map_img, dpi=250, bbox_inches="tight")
    except Exception:
        plt.close(fig)
    finally:
        plt.close(fig)

    try:
        pdf.image(map_img, x=MAP_X, y=MAP_Y, w=MAP_W, h=MAP_H)
    except Exception:
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 8, "Map image failed to render.", ln=1, align="C")

    # legend
    legend_y = MAP_Y + MAP_H + LEGEND_GAP
    pdf.set_y(legend_y)
    pdf.set_fill_color(245, 245, 240)
    pdf.set_draw_color(180, 180, 180)
    pdf.rect(MAP_X, legend_y, MAP_W, 40, style="FD")
    pdf.set_font("Helvetica", "", 11)
    col1 = [
        f"Range: {user_inputs.get('range_name','')}",
        f"RF: {user_inputs.get('rf_name','')}",
        f"Beat: {user_inputs.get('beat_name','')}",
        f"Year: {user_inputs.get('year_of_work','')}",
    ]
    col2 = [
        f"Density: {density}",
        f"Area of Invasive: {area_invasive} Ha",
        f"Cell Size: {cell_size} m",
        f"Overlay: {'Yes' if overlay_gdf is not None and not overlay_gdf.empty else 'No'}",
    ]
    for i in range(4):
        pdf.text(MAP_X + 10, legend_y + 10 + i * 6, col1[i])
        pdf.text(MAP_X + 100, legend_y + 10 + i * 6, col2[i])

    pdf.set_y(legend_y + 47)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 5, "Note: Satellite background (Esri) and boundaries are automatically generated.")
    pdf.set_text_color(0, 0, 0)

    # Page 2 - GPS table
    pdf.add_page()
    if overlay_gdf is not None and not overlay_gdf.empty:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Corner GPS of Overlay Area", ln=1, align="C")
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(25, 8, "S.No", 1, align="C")
        pdf.cell(75, 8, "Latitude", 1, align="C")
        pdf.cell(75, 8, "Longitude", 1, align="C")
        pdf.ln(8)
        pdf.set_font("Helvetica", "", 10)
        row = 1
        overlay = overlay_gdf.to_crs(4326)
        for geom in overlay.geometry:
            if geom.is_empty:
                continue
            coords = []
            if geom.geom_type == "Polygon":
                coords = list(geom.exterior.coords)
            elif geom.geom_type == "MultiPolygon":
                for part in geom.geoms:
                    coords.extend(list(part.exterior.coords))
            for lon, lat, *_ in coords:
                pdf.cell(25, 7, str(row), 1, align="C")
                pdf.cell(75, 7, f"{lat:.6f}", 1, align="C")
                pdf.cell(75, 7, f"{lon:.6f}", 1, align="C")
                pdf.ln(7)
                row += 1
                if pdf.get_y() > 240:
                    pdf.add_page()
                    pdf.set_font("Helvetica", "B", 11)
                    pdf.cell(25, 8, "S.No", 1, align="C")
                    pdf.cell(75, 8, "Latitude", 1, align="C")
                    pdf.cell(75, 8, "Longitude", 1, align="C")
                    pdf.ln(8)
                    pdf.set_font("Helvetica", "", 10)

    # Invasive Grid Map Box (try place on same page)
    if overlay_gdf is not None and not overlay_gdf.empty:
        df_overlay, overlay_area_ha, total_grid_area_ha = compute_overlay_area_by_grid(cells_ll, overlay_gdf)
        overlay_union = unary_union(overlay_gdf.geometry)
        clipped_rows = []
        for i, cell in enumerate(cells_ll, start=1):
            inter = cell.intersection(overlay_union)
            if inter.is_empty:
                continue
            clipped_rows.append({"grid_id": int(i), "geometry": inter})

        if clipped_rows:
            clipped_gdf = gpd.GeoDataFrame(clipped_rows, geometry=[r['geometry'] for r in clipped_rows], crs="EPSG:4326")
            clipped_gdf['label_pt'] = clipped_gdf.geometry.representative_point()

            tmp_dir = tempfile.gettempdir()
            invasive_map_img = os.path.join(tmp_dir, "map_invasive.png")
            fig, ax = plt.subplots(figsize=(7, 5.8))
            try:
                clipped_3857 = clipped_gdf.to_crs(3857)
                overlay_3857 = overlay_gdf.to_crs(3857)
                clipped_3857.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=1)
                overlay_3857.boundary.plot(ax=ax, color="#FFD700", linewidth=3)
                try:
                    ctx.add_basemap(ax, crs=3857, source=ctx.providers.Esri.WorldImagery, attribution=False)
                except Exception:
                    pass
                ax.axis('off')
                for _, crow in clipped_3857.iterrows():
                    try:
                        pt = crow.geometry.representative_point()
                        gid = int(crow['grid_id']) if 'grid_id' in crow.index else ''
                        ax.text(pt.x, pt.y, str(gid), fontsize=8, ha='center', va='center', color='#03fcfc')
                    except Exception:
                        pass
                plt.tight_layout(pad=0.1)
                fig.savefig(invasive_map_img, dpi=250, bbox_inches='tight')
            except Exception:
                pass
            finally:
                plt.close(fig)

            y_now = pdf.get_y()
            page_bottom_limit = 297 - 15
            if y_now + MAP_H + 20 > page_bottom_limit:
                pdf.add_page()

            pdf.ln(10)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Invasive Grid Fall", ln=1, align="C")
            pdf.ln(2)
            try:
                pdf.image(invasive_map_img, x=MAP_X, y=pdf.get_y(), w=MAP_W, h=MAP_H)
            except Exception:
                pdf.set_font("Helvetica", "I", 10)
                pdf.cell(0, 8, "Invasive map image failed to render.", ln=1, align="C")
            pdf.set_y(pdf.get_y() + MAP_H + 4)
        else:
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 8, "No grid cells intersect the overlay — map not generated.", ln=1)

    # Second table: Grid ID | Area inside overlay
    if overlay_gdf is not None and not overlay_gdf.empty:
        try:
            if 'df_overlay' not in locals():
                df_overlay, overlay_area_ha, total_grid_area_ha = compute_overlay_area_by_grid(cells_ll, overlay_gdf)

            rows_needed = len(df_overlay) if df_overlay is not None else 0
            est_height = rows_needed * 8 + 40
            if pdf.get_y() + est_height > page_bottom_limit:
                pdf.add_page()

            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, "Grid Area Inside Overlay Boundary (Detail)", ln=1, align="C")
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(30, 8, "Grid ID", 1, align="C")
            pdf.cell(80, 8, "Area Inside Overlay (Ha)", 1, align="C")
            pdf.ln(8)

            pdf.set_font("Helvetica", "", 11)

            if df_overlay is not None and not df_overlay.empty:
                for idx, row in df_overlay.iterrows():
                    pdf.cell(30, 8, str(int(row["grid_id"])), 1, align="C")
                    pdf.cell(80, 8, f"{row['intersection_area_ha']:.4f}", 1, align="C")
                    pdf.ln(8)
                    if pdf.get_y() > 240:
                        pdf.add_page()
                        pdf.set_font("Helvetica", "B", 11)
                        pdf.cell(30, 8, "Grid ID", 1, align="C")
                        pdf.cell(80, 8, "Area Inside Overlay (Ha)", 1, align="C")
                        pdf.ln(8)
                        pdf.set_font("Helvetica", "", 11)
            else:
                pdf.cell(0, 8, "No intersecting grid cells.", ln=1)

            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, f"TOTAL AREA INSIDE OVERLAY: {math.ceil(total_grid_area_ha)} Ha", ln=1, align="L")
        except Exception as _e:
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 8, f"Overlay detail table generation failed: {_e}", ln=1)

    # QR and upload
    y_pos = pdf.get_y()
    try:
        live_kml_url = "https://krishnaSureshFor.github.io/tnforest_kml_to_grid_v2.0"
        qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H)
        qr.add_data(live_kml_url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        y_pos = pdf.get_y() + 12
        if y_pos > 230:
            pdf.add_page()
            y_pos = 30
        pdf.set_font("Helvetica", "B", 11)
        pdf.text(20, y_pos, "Scan QR to View KML File:")
        pdf.image(buf, x=20, y=y_pos + 5, w=38, type="PNG")
        pdf.set_draw_color(0, 0, 0)
        pdf.rect(19, y_pos + 4, 40, 40)
        pdf.set_xy(20, y_pos + 47)
        pdf.set_font("Helvetica", "U", 9)
        pdf.set_text_color(0, 0, 255)
        pdf.cell(0, 8, live_kml_url, link=live_kml_url)
        pdf.set_text_color(0, 0, 0)
    except Exception:
        pass

    result = pdf.output(dest="S")
    return bytes(result) if isinstance(result, (bytes, bytearray)) else result.encode("latin1", errors="ignore")


# -------------------------------
# Streamlit UI - Clean rebuild
# -------------------------------

st.title("TN Forest — KML to Grid (v3)")

# Sidebar inputs
st.sidebar.header("Inputs")
uploaded_aoi = st.sidebar.file_uploader("Upload RF / AOI KML or KMZ", type=["kml", "kmz"], key="aoi_upload")
uploaded_overlay = st.sidebar.file_uploader("Upload Overlay KML (optional)", type=["kml", "kmz"], key="overlay_upload")
cell_size = st.sidebar.number_input("Cell size (m)", value=100, step=10)
density = st.sidebar.text_input("Density (e.g. low/med/high)")
area_invasive = st.sidebar.number_input("Area of invasive (Ha)", value=0.0)

user_inputs = {
    "range_name": st.sidebar.text_input("Range Name", value=""),
    "rf_name": st.sidebar.text_input("RF Name", value=""),
    "beat_name": st.sidebar.text_input("Beat Name", value=""),
    "year_of_work": st.sidebar.text_input("Year of Work", value="")
}

col1, col2 = st.columns(2)
with col1:
    st.header("AOI / Overlay")
    if uploaded_aoi:
        try:
            aoi_gdf = gpd.read_file(uploaded_aoi)
            st.session_state["aoi_gdf"] = aoi_gdf
            st.success("AOI uploaded")
        except Exception as e:
            st.error(f"Failed to read AOI: {e}")
    else:
        aoi_gdf = st.session_state.get("aoi_gdf")

    if uploaded_overlay:
        try:
            overlay_gdf = gpd.read_file(uploaded_overlay)
            st.session_state["overlay_gdf"] = overlay_gdf
            st.success("Overlay uploaded")
        except Exception as e:
            st.error(f"Failed to read overlay: {e}")
    else:
        overlay_gdf = st.session_state.get("overlay_gdf")

with col2:
    st.header("Actions")
    if st.button("Generate Grid"):
        # Placeholder grid generation: create simple square grid covering AOI bounds
        if "aoi_gdf" not in st.session_state or st.session_state.get("aoi_gdf") is None:
            st.warning("Upload AOI first")
        else:
            aoi = st.session_state["aoi_gdf"]
            bounds = aoi.total_bounds  # minx, miny, maxx, maxy
            minx, miny, maxx, maxy = bounds[0], bounds[1], bounds[2], bounds[3]
            nx = max(1, int(((maxx - minx) * 111320) // cell_size))
            ny = max(1, int(((maxy - miny) * 110540) // cell_size))
            cells = []
            # naive grid in lat/lon (approx) — for production use accurate projected grid
            lon_step = (maxx - minx) / max(1, nx)
            lat_step = (maxy - miny) / max(1, ny)
            gid = 1
            from shapely.geometry import Polygon
            for i in range(nx):
                for j in range(ny):
                    lx = minx + i * lon_step
                    ly = miny + j * lat_step
                    poly = Polygon([(lx, ly), (lx + lon_step, ly), (lx + lon_step, ly + lat_step), (lx, ly + lat_step)])
                    cells.append(poly)
                    gid += 1
            st.session_state["cells_ll"] = cells
            st.session_state["merged_ll"] = aoi.unary_union
            st.session_state["labeled_kml"] = None
            st.session_state["pdf_bytes"] = None
            st.session_state["generated"] = True
            st.success("Grid generated — scroll down for outputs")

# Edit AOI workflow
if st.session_state.get("aoi_gdf") is not None:
    if st.button("Change RF Kml (Edit Boundary)"):
        st.session_state["edit_mode"] = True
        st.experimental_rerun()

# Map preview / edit page
if st.session_state.get("edit_mode", False):
    st.title("Edit AOI Boundary")
    aoi_gdf = st.session_state.get("aoi_gdf")
    if LEAFMAP_AVAILABLE:
        m = leafmap.Map(center=[aoi_gdf.geometry.centroid.y.mean(), aoi_gdf.geometry.centroid.x.mean()], zoom=13)
        m.add_gdf(aoi_gdf, layer_name="AOI")
        m.add_draw_control(draw_options={"polyline": False, "polygon": True, "circle": False, "rectangle": True, "marker": False}, edit_options={"poly": {"allowIntersection": False}})
        m.to_streamlit(height=600)
        edited_file = st.file_uploader("Upload Edited AOI (KML/KMZ)", type=["kml", "kmz"])
        if edited_file:
            try:
                edited_gdf = gpd.read_file(edited_file)
                st.session_state["aoi_gdf"] = edited_gdf
                st.success("Edited AOI uploaded")
                if st.button("Re-run Grid Formation"):
                    st.session_state["generated"] = False
                    st.session_state["edit_mode"] = False
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to read edited file: {e}")
    else:
        st.info("Live edit requires 'leafmap' — upload edited KML here instead.")
        edited_file = st.file_uploader("Upload Edited AOI (KML/KMZ)", type=["kml", "kmz"])
        if edited_file:
            try:
                edited_gdf = gpd.read_file(edited_file)
                st.session_state["aoi_gdf"] = edited_gdf
                st.success("Edited AOI uploaded")
                if st.button("Re-run Grid Formation"):
                    st.session_state["generated"] = False
                    st.session_state["edit_mode"] = False
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to read edited file: {e}")
    # Download modified AOI
    if st.session_state.get("aoi_gdf") is not None:
        tmp = tempfile.mkdtemp()
        kml_out = os.path.join(tmp, "Modified_AOI.kml")
        try:
            st.session_state["aoi_gdf"].to_file(kml_out, driver="KML")
            with open(kml_out, "rb") as f:
                styled_download_button("Download Modified KML", f.read(), "Modified_AOI.kml", "application/vnd.google-earth.kml+xml", icon="📥", bg="#0ea5e9")
        except Exception:
            st.error("Failed to prepare Modified KML for download")
    st.stop()

# Main outputs section
if st.session_state.get("generated", False):
    st.success("Grid generated — outputs below")
    cells_ll = st.session_state.get("cells_ll", [])
    merged_ll = st.session_state.get("merged_ll")
    overlay_gdf = st.session_state.get("overlay_gdf")

    # Build labeled KML (simple — grid id labels)
    try:
        tmp = tempfile.mkdtemp()
        labeled_kml_path = os.path.join(tmp, "labeled_grid.kml")
        gdf_k = gpd.GeoDataFrame({"grid_id": range(1, len(cells_ll) + 1)}, geometry=cells_ll, crs="EPSG:4326")
        gdf_k.to_file(labeled_kml_path, driver="KML")
        with open(labeled_kml_path, "rb") as f:
            labeled_kml_bytes = f.read()
        st.session_state["labeled_kml"] = labeled_kml_bytes
    except Exception:
        st.session_state["labeled_kml"] = None

    # Build PDF
    try:
        pdf_bytes = build_pdf_report_standard(cells_ll, merged_ll, user_inputs, cell_size, overlay_gdf, st.sidebar.text_input("Report Title", value=""), density, area_invasive, labeled_kml=st.session_state.get("labeled_kml"))
        st.session_state["pdf_bytes"] = pdf_bytes
    except Exception as e:
        st.error(f"PDF generation failed: {e}")

    st.markdown("### Map Preview")
    colmap1, colmap2 = st.columns([2,1])
    with colmap1:
        st.write("Map of AOI + Grid + Overlay")
        # Static map image
        try:
            tmp_img = os.path.join(tempfile.gettempdir(), "map_overlay.png")
            st.image(tmp_img, use_column_width=True)
        except Exception:
            st.info("Map image not available")

    # Downloads (styled)
    st.markdown("### 💾 Downloads (Styled)")
    c1, c2, c3 = st.columns(3)
    with c1:
        # Grid Only KML
        try:
            grid_only_kml = gpd.GeoDataFrame(geometry=cells_ll, crs="EPSG:4326")
            kml_path = os.path.join(tempfile.gettempdir(), "grid_only.kml")
            grid_only_kml.to_file(kml_path, driver="KML")
            with open(kml_path, "rb") as f:
                styled_download_button("Download Grid Only KML", f.read(), "grid_only.kml", "application/vnd.google-earth.kml+xml", icon="🗺️", bg="#0284c7")
        except Exception:
            st.error("Preparing grid KML failed")
    with c2:
        # Labeled KML
        if st.session_state.get("labeled_kml"):
            styled_download_button("Download Labeled + Overlay KML", st.session_state["labeled_kml"], "merged_labeled.kml", "application/vnd.google-earth.kml+xml", icon="🧾", bg="#16a34a")
    with c3:
        if st.session_state.get("pdf_bytes"):
            styled_download_button("Download Invasive Report (PDF)", st.session_state.get("pdf_bytes"), "Invasive_Report.pdf", "application/pdf", icon="📄", bg="#f97316")

    # Shapefile downloads: Grid with ID & Grid Only
    st.markdown("### 📦 Shapefile Downloads")
    c4, c5 = st.columns(2)
    try:
        gdf_grid_id = gpd.GeoDataFrame({"grid_id": range(1, len(cells_ll) + 1)}, geometry=cells_ll, crs="EPSG:4326")
        gdf_grid_only = gpd.GeoDataFrame(geometry=cells_ll, crs="EPSG:4326")
        shp_grid_id = export_shapefile_zip(gdf_grid_id, "Grid_with_ID")
        shp_grid_only = export_shapefile_zip(gdf_grid_only, "Grid_Only")
        with c4:
            styled_download_button("Grid (Shapefile + ID)", shp_grid_id, "Grid_with_ID.zip", "application/zip", icon="🗂️", bg="#10b981")
        with c5:
            styled_download_button("Grid Only (Shapefile)", shp_grid_only, "Grid_Only.zip", "application/zip", icon="📁", bg="#f59e0b")
    except Exception:
        st.error("Shapefile export failed")

    st.markdown("---")
    st.info("If you edited AOI, use the 'Change RF Kml' button to go back to edit mode.")

else:
    st.info("👆 Use Sidebar to upload AOI (KML/KMZ) and optionally Overlay, set parameters, then click \"Generate Grid\".")

# End of file
""
