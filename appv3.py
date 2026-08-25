# appv3.py (patched)
# Patched by assistant: single shapefile download for Grid+Overlay & Grid Only,
# Grid ID color in PDF (#03fcfc), invasive map labels colored #03fcfc.

try:
    import leafmap.foliumap as leafmap
    LEAFMAP_AVAILABLE = True
except Exception:
    LEAFMAP_AVAILABLE = False
import streamlit as st
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
from shapely.geometry import mapping
from pyproj import CRS
import math, os, tempfile, zipfile, shutil
from streamlit_folium import st_folium
import folium
from fpdf import FPDF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import contextily as ctx
from lxml import etree
import fiona
import uuid, qrcode
from io import BytesIO
import base64
from PIL import Image

def save_kml_for_viewer(kml_text):
    """Save KML with unique ID to public_kml folder for permanent hosting."""
    kml_id = str(uuid.uuid4())[:8]  # short unique ID
    out_dir = os.path.join("public_kml")
    os.makedirs(out_dir, exist_ok=True)
    kml_path = os.path.join(out_dir, f"{kml_id}.kml")
    with open(kml_path, "w", encoding="utf-8") as f:
        f.write(kml_text)
    return kml_id, kml_path


def make_qr_code(url):
    """Return QR code image (as BytesIO)."""
    qr = qrcode.QRCode(box_size=3, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    # --- Prepare exact physical size (~300 px = 40 mm @ 190 DPI) ---
    target_px = 300
    img = img.resize((target_px, target_px), Image.NEAREST)
    
    # --- Save crisp 8-bit RGB PNG ---
    buf = BytesIO()
    img.convert("RGB").save(buf, format="PNG", dpi=(190, 190))
    buf.seek(0)
    
    # NOTE: this helper is not used directly in the current PDF pipeline.
    return buf

# ================================================================
# APP CONFIG + THEME
# ================================================================
st.set_page_config(page_title="KML Grid Generator v2.0", layout="wide")

# 🌳 Custom gradient background and theme
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #f1f8e9 0%, #e8f5e9 100%); }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2e4c16 0%, #172a08 100%);
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div[data-baseweb="select"] * {
    color: #fdf5c9 !important;
}
section[data-testid="stSidebar"] h2 {
    color: #ffeb3b !important; font-weight: 800 !important; text-align: center;
    border-bottom: 2px solid #ffeb3b; padding-bottom: 6px;
}
input, textarea, select {
    background-color: #f1f8e9 !important; border: 1px solid #558b2f !important;
    color: #172a08 !important; border-radius: 6px !important;
}
div.stButton > button {
    background: linear-gradient(90deg, #fbc02d, #ffeb3b);
    color: #172a08 !important; font-weight: 700; border-radius: 10px; border: none;
    box-shadow: 1px 2px 5px rgba(0,0,0,0.2); transition: all 0.2s ease;
}
div.stButton > button:hover { background: linear-gradient(90deg, #ffeb3b, #fff59d); transform: scale(1.03); }
.stDownloadButton > button {
    background: linear-gradient(90deg, #33691e, #558b2f);
    color: #ffeb3b !important; border-radius: 10px; border: none; font-weight: 700;
    box-shadow: 1px 2px 4px rgba(0,0,0,0.2); transition: all 0.2s ease;
}
.stDownloadButton > button:hover { background: linear-gradient(90deg, #558b2f, #689f38); transform: scale(1.03); }
iframe[title="streamlit_folium"] {
    border-radius: 18px;
    border: 5px double transparent;
    background-image: linear-gradient(white, white), linear-gradient(90deg, #33691e, #ffeb3b);
    background-origin: border-box; background-clip: content-box, border-box;
    box-shadow: 0 5px 12px rgba(0,0,0,0.25); padding: 2px;
}
/* Ensure the text inside the file uploader drop zone is visible */
.stFileUploader > div > div > div > div > div > p {
    color: #fdf5c9 !important;
}
/* File uploader 'Browse files' text color fix */
.stFileUploader button {
    color: #172a08 !important;
}
</style>
""", unsafe_allow_html=True)

# ================================================================
# HEADER
# ================================================================
st.markdown("""
<div style='text-align:center; padding:15px; 
background:linear-gradient(90deg, #2e4c16, #558b2f);
border-radius:10px; color:#ffeb3b; font-size:28px; font-weight:700;
box-shadow:0 4px 10px rgba(0,0,0,0.3); letter-spacing:1px;'>
🌿 KML Grid Generator v2.0 - Unified Invasive Report 🌿
</div>
""", unsafe_allow_html=True)

# ================================================================
# SIDEBAR
# ================================================================
st.sidebar.header("⚙️ Tool Settings")

with st.sidebar.expander("📂 Upload Files (AOI / Overlay)", expanded=True):
    uploaded_aoi = st.file_uploader("Upload RF/RL kml", type=["kml", "kmz"], key="aoi_file")
    overlay_file = st.file_uploader("Upload Overlay Kml", type=["kml", "kmz"], key="overlay_file")


with st.sidebar.expander("🌲 KML Label Details"):
    range_name = st.text_input("Range Name", placeholder="Range Name", key="range_name")
    rf_name = st.text_input("RF Name", placeholder="RF/RL", key="rf_name")
    beat_name = st.text_input("Beat Name", placeholder="Beat Name", key="beat_name")
    year_of_work = st.text_input("Year of Work", placeholder="Year of Work", key="year_of_work")

with st.sidebar.expander("📄 PDF Report Details"):
    title_text = st.text_input("Report Title", placeholder="Title, Range", key="title_text")
    density = st.text_input("Density", placeholder="Medium/Light/Dense", key="density")
    area_invasive = st.text_input("Area of Invasive (Ha)", placeholder="Area in Ha", key="area_invasive")
    cell_size = st.number_input("Grid Cell Size (m)", 10, 2000, 100, 10, key="cell_size")
    generate_pdf = st.checkbox("Generate PDF Report", value=True, key="generate_pdf")

col1, col2 = st.sidebar.columns(2)
with col1: generate_click = st.button("▶ Generate Grid", key="btn_generate")
with col2: reset_click = st.button("🔄 Reset Map", key="btn_reset")

# ================================================================
# STATE
# ================================================================
def init_state():
    if "user_inputs" not in st.session_state:
        st.session_state["user_inputs"] = {
            "range_name": range_name, "rf_name": rf_name,
            "beat_name": beat_name, "year_of_work": year_of_work
        }
    if "generated" not in st.session_state:
        st.session_state["generated"] = False

init_state()

if reset_click:
    st.session_state.clear()
    init_state()
    st.rerun()
if generate_click:
    st.session_state["user_inputs"] = {
        "range_name": range_name, "rf_name": rf_name,
        "beat_name": beat_name, "year_of_work": year_of_work
    }
    st.session_state["generated"] = True

# ================================================================
# HELPERS
# ================================================================
def read_kml_kmz(path):
    """Robustly read KML/KMZ including multi-layer/multi-feature files."""
    import pandas as pd
    gdfs = []
    
    def read_layers(kml_path):
        res = []
        try:
            layers = fiona.listlayers(kml_path)
            # If no layers, try reading without layer spec (old behavior fallback)
            if not layers:
                try:
                    res.append(gpd.read_file(kml_path))
                except Exception: pass
            for l in layers:
                try:
                    # Explicitly use KML driver if possible
                    rdf = gpd.read_file(kml_path, layer=l)
                    if rdf is not None and not rdf.empty:
                        if rdf.crs is None:
                            rdf.set_crs("EPSG:4326", inplace=True)
                        res.append(rdf.to_crs("EPSG:4326"))
                except Exception:
                    continue
        except Exception:
            # Fallback if listlayers fails
            try:
                rdf = gpd.read_file(kml_path)
                if rdf.crs is None:
                    rdf.set_crs("EPSG:4326", inplace=True)
                res.append(rdf.to_crs("EPSG:4326"))
            except Exception: pass
        return res

    if str(path).lower().endswith(".kmz"):
        try:
            with zipfile.ZipFile(path, 'r') as z:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    z.extractall(tmp_dir)
                    for root, _, files in os.walk(tmp_dir):
                        for file in files:
                            if file.lower().endswith(".kml"):
                                gdfs.extend(read_layers(os.path.join(root, file)))
        except Exception:
             # If zip fails, try reading as raw
             gdfs.extend(read_layers(path))
    else:
        gdfs.extend(read_layers(path))
        
    if not gdfs:
        return None
        
    # Merge and ensure consistent CRS
    final_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")
    return final_gdf

def read_kml_safely(path):
    """Robustly read KML using the improved KML/KMZ helper."""
    return read_kml_kmz(path)

def clean_polygon_gdf(gdf: gpd.GeoDataFrame | None) -> gpd.GeoDataFrame | None:
    """
    Keep only valid Polygon / MultiPolygon geometries.
    This avoids Folium 'coordinates' KeyError for non-polygon/empty features.
    """
    if gdf is None or gdf.empty:
        return gdf
    gdf = gdf[gdf.geometry.notnull()]
    gdf = gdf[~gdf.geometry.is_empty]
    gdf = gdf[gdf.geometry.is_valid]
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    return gdf

def utm_crs_for_lonlat(lon, lat):
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def compute_overlay_area_by_grid(cells_ll, overlay_gdf):
    """Return df_overlay, overlay_area_ha, total_grid_area_ha"""
    import pandas as pd
    from shapely.ops import unary_union
    if overlay_gdf is None or overlay_gdf.empty:
        return pd.DataFrame([], columns=["grid_id","intersection_area_ha"]), 0.0, 0.0

    overlay_union = unary_union(overlay_gdf.geometry)
    centroid = overlay_union.centroid
    utm_overlay = utm_crs_for_lonlat(centroid.x, centroid.y)
    overlay_area_ha = (
        gpd.GeoSeries([overlay_union], crs="EPSG:4326")
        .to_crs(utm_overlay).area.iloc[0] / 10000.0
    )

    rows = []
    for i, cell in enumerate(cells_ll, start=1):
        inter = cell.intersection(overlay_union)
        if inter.is_empty:
            continue
        c_centroid = cell.centroid
        utm = utm_crs_for_lonlat(c_centroid.x, c_centroid.y)
        inter_area_ha = (
            gpd.GeoSeries([inter], crs="EPSG:4326")
            .to_crs(utm).area.iloc[0] / 10000.0
        )
        rows.append({"grid_id": int(i), "intersection_area_ha": round(inter_area_ha, 4)})
    df = pd.DataFrame(rows)
    total_grid_area_ha = df["intersection_area_ha"].sum() if not df.empty else 0.0
    return df, overlay_area_ha, total_grid_area_ha

def make_grid_exact_clipped(polygons_ll, cell_size_m=100):
    """Generate clipped grid cells (ensures valid non-empty geometries in EPSG:4326)."""
    merged_ll = unary_union(polygons_ll)

    # --- Reproject AOI to local UTM ---
    centroid = merged_ll.centroid
    utm = utm_crs_for_lonlat(centroid.x, centroid.y)
    merged_utm = gpd.GeoSeries([merged_ll], crs="EPSG:4326").to_crs(utm)

    minx, miny, maxx, maxy = merged_utm.total_bounds
    cols = int(math.ceil((maxx - minx) / cell_size_m))
    rows = int(math.ceil((maxy - miny) / cell_size_m))

    cells = []
    aoi_union = merged_utm.unary_union

    for i in range(cols):
        for j in range(rows):
            x0, y0 = minx + i * cell_size_m, miny + j * cell_size_m
            cell = box(x0, y0, x0 + cell_size_m, y0 + cell_size_m)
            inter = cell.intersection(aoi_union)
            # --- Filter valid polygons only ---
            if not inter.is_empty and inter.is_valid and inter.area > 0:
                cells.append(inter)

    # --- Reproject final grid cells back to EPSG:4326 for KML ---
    if not cells:
        print("⚠️ No valid grid cells generated — check AOI extent.")
    cells_ll = [
        gpd.GeoSeries([c], crs=utm).to_crs(epsg=4326).iloc[0]
        for c in cells if not c.is_empty
    ]

    # --- Also ensure merged_ll is lat/lon ---
    merged_ll = gpd.GeoSeries([merged_ll], crs=4326).to_crs(epsg=4326).iloc[0]

    return cells_ll, merged_ll

# ================================================================
# KML GENERATORS (with Description + Balloon Popups)
# ================================================================
def _ring_coords_to_kml(ring):
    return " ".join(f"{pt[0]},{pt[1]},0" for pt in ring.coords if len(pt) >= 2)

def _write_polygon_coords(ns, parent_polygon_elem, geom):
    def write_one(poly):
        outer = etree.SubElement(parent_polygon_elem, "{%s}outerBoundaryIs" % ns)
        lr_out = etree.SubElement(outer, "{%s}LinearRing" % ns)
        etree.SubElement(lr_out, "{%s}coordinates" % ns).text = _ring_coords_to_kml(poly.exterior)
    if geom.geom_type == "Polygon":
        write_one(geom)
    elif geom.geom_type == "MultiPolygon":
        for part in geom.geoms:
            poly_elem = etree.SubElement(parent_polygon_elem.getparent(), "{%s}Polygon" % ns)
            write_one(part)

def _make_grid_balloon_text(user_inputs):
    return (
        "<![CDATA["
        "<b>Grid ID:</b> $[name]<br>"
        f"<b>Range:</b> {user_inputs['range_name']}<br>"
        f"<b>RF/RL:</b> {user_inputs['rf_name']}<br>"
        f"<b>Beat:</b> {user_inputs['beat_name']}<br>"
        f"<b>Year:</b> {user_inputs['year_of_work']}<br>"
        "<b>Area (Ha):</b> $[area_ha]<br>"
    )

def generate_grid_only_kml(cells_ll, merged_ll, user_inputs):
    """Grid-only KML with same popup label as merged (no overlay)."""
    ns = "http://www.opengis.net/kml/2.2"
    kml = etree.Element("{%s}kml" % ns)
    doc = etree.SubElement(kml, "{%s}Document" % ns)

    etree.SubElement(doc, "{%s}name" % ns).text = "Grid Only"
    etree.SubElement(doc, "{%s}description" % ns).text = (
        "Grid-only file with labeled cells for field use. "
        "Developed by Krishna."
    )

    style_grid = etree.SubElement(doc, "{%s}Style" % ns, id="gridStyle")
    ls1 = etree.SubElement(style_grid, "{%s}LineStyle" % ns)
    etree.SubElement(ls1, "{%s}color" % ns).text = "ff0000ff"  # red
    etree.SubElement(ls1, "{%s}width" % ns).text = "1"
    ps1 = etree.SubElement(style_grid, "{%s}PolyStyle" % ns)
    etree.SubElement(ps1, "{%s}fill" % ns).text = "0"
    balloon = etree.SubElement(style_grid, "{%s}BalloonStyle" % ns)
    etree.SubElement(balloon, "{%s}text" % ns).text = _make_grid_balloon_text(user_inputs)

    for i, cell in enumerate(cells_ll, 1):
        centroid = cell.centroid
        utm_crs = utm_crs_for_lonlat(centroid.x, centroid.y)
        area_ha = gpd.GeoSeries([cell], crs=4326).to_crs(utm_crs).area.iloc[0] / 10000.0

        pm = etree.SubElement(doc, "{%s}Placemark" % ns)
        etree.SubElement(pm, "{%s}name" % ns).text = str(i)
        etree.SubElement(pm, "{%s}styleUrl" % ns).text = "#gridStyle"

        ext_data = etree.SubElement(pm, "{%s}ExtendedData" % ns)
        d = etree.SubElement(ext_data, "{%s}Data" % ns, name="area_ha")
        etree.SubElement(d, "{%s}value" % ns).text = f"{area_ha:.2f}"

        desc = etree.SubElement(pm, "{%s}description" % ns)
        desc.text = f"Grid {i} — Area: {area_ha:.2f} ha"

        poly = etree.SubElement(pm, "{%s}Polygon" % ns)
        _write_polygon_coords(ns, poly, cell)

    return etree.tostring(kml, pretty_print=True, xml_declaration=True, encoding="UTF-8").decode("utf-8")

def generate_labeled_kml(cells_ll, merged_ll, user_inputs, overlay_gdf=None):
    """Labeled grid + overlay (gold) with popups and description."""
    ns = "http://www.opengis.net/kml/2.2"
    kml = etree.Element("{%s}kml" % ns)
    doc = etree.SubElement(kml, "{%s}Document" % ns)

    etree.SubElement(doc, "{%s}name" % ns).text = "Labeled Grid + Overlay"
    etree.SubElement(doc, "{%s}description" % ns).text = (
        "Developed by Krishna"
    )

    # Grid style
    style_grid = etree.SubElement(doc, "{%s}Style" % ns, id="gridStyle")
    ls1 = etree.SubElement(style_grid, "{%s}LineStyle" % ns)
    etree.SubElement(ls1, "{%s}color" % ns).text = "ff0000ff"  # red
    etree.SubElement(ls1, "{%s}width" % ns).text = "1"
    ps1 = etree.SubElement(style_grid, "{%s}PolyStyle" % ns)
    etree.SubElement(ps1, "{%s}fill" % ns).text = "0"
    balloon = etree.SubElement(style_grid, "{%s}BalloonStyle" % ns)
    etree.SubElement(balloon, "{%s}text" % ns).text = _make_grid_balloon_text(user_inputs)

    # Overlay style (golden yellow 3px)
    style_overlay = etree.SubElement(doc, "{%s}Style" % ns, id="overlayStyle")
    ls2 = etree.SubElement(style_overlay, "{%s}LineStyle" % ns)
    etree.SubElement(ls2, "{%s}color" % ns).text = "ff00d7ff"  # ABGR for #FFD700
    etree.SubElement(ls2, "{%s}width" % ns).text = "3"
    ps2 = etree.SubElement(style_overlay, "{%s}PolyStyle" % ns)
    etree.SubElement(ps2, "{%s}fill" % ns).text = "0"

    # Grid placemarks
    for i, cell in enumerate(cells_ll, 1):
        centroid = cell.centroid
        utm_crs = utm_crs_for_lonlat(centroid.x, centroid.y)
        area_ha = gpd.GeoSeries([cell], crs=4326).to_crs(utm_crs).area.iloc[0] / 10000.0

        pm = etree.SubElement(doc, "{%s}Placemark" % ns)
        etree.SubElement(pm, "{%s}name" % ns).text = str(i)
        etree.SubElement(pm, "{%s}styleUrl" % ns).text = "#gridStyle"

        ext_data = etree.SubElement(pm, "{%s}ExtendedData" % ns)
        d = etree.SubElement(ext_data, "{%s}Data" % ns, name="area_ha")
        etree.SubElement(d, "{%s}value" % ns).text = f"{area_ha:.2f}"

        desc = etree.SubElement(pm, "{%s}description" % ns)
        desc.text = f"Grid {i} — Area: {area_ha:.2f} ha"

        poly = etree.SubElement(pm, "{%s}Polygon" % ns)
        _write_polygon_coords(ns, poly, cell)

    # Overlay boundary
    if overlay_gdf is not None and not overlay_gdf.empty:
        og = overlay_gdf.to_crs(4326)
        for geom in og.geometry:
            if geom.is_empty:
                continue
            pm = etree.SubElement(doc, "{%s}Placemark" % ns)
            etree.SubElement(pm, "{%s}name" % ns).text = "Overlay Boundary"
            etree.SubElement(pm, "{%s}styleUrl" % ns).text = "#overlayStyle"
            poly = etree.SubElement(pm, "{%s}Polygon" % ns)
            _write_polygon_coords(ns, poly, geom)

    return etree.tostring(kml, pretty_print=True, xml_declaration=True, encoding="UTF-8").decode("utf-8")

# ================================================================
# PDF REPORT FUNCTION (stable layout + correct footer)
# ================================================================
def build_pdf_report_standard(
    cells_ll, merged_ll, user_inputs, cell_size,
    overlay_gdf, title_text, density, area_invasive, labeled_kml=None
):
    import geopandas as gpd, matplotlib.pyplot as plt, contextily as ctx, tempfile, os
    from fpdf import FPDF
    import qrcode
    import uuid
    from io import BytesIO
    from github import Github
    from PIL import Image
    import base64

    MAP_X, MAP_Y, MAP_W, MAP_H, LEGEND_GAP = 15, 55, 180, 145, 8
    EMBLEM_PATH = os.path.join(os.path.dirname(__file__), "tn_emblem.png")

    # -------------------------------
    # Helper: Push KML file to GitHub
    # -------------------------------
    def push_kml_to_repo(kml_path, kml_id, repo_name="krishnaSureshFor/tnforest_kml_to_grid_v2.0"):
        """Push generated KML to GitHub repo's public_kml folder."""
        token = os.getenv("GITHUB_TOKEN")  # set this in Streamlit Cloud → Settings → Secrets
        print("🔍 GITHUB_TOKEN found:", bool(token))
        if not token:
            print("⚠️ Missing GITHUB_TOKEN — cannot push to repo")
            return None
        try:
            g = Github(token)
            repo = g.get_repo(repo_name)
            with open(kml_path, "r", encoding="utf-8") as f:
                content = f.read()
            path_in_repo = f"public_kml/{kml_id}.kml"

            # If file already exists, update instead of recreate
            try:
                existing = repo.get_contents(path_in_repo, ref="main")
                repo.update_file(
                    existing.path, f"Update KML {kml_id}", content,
                    existing.sha, branch="main"
                )
            except Exception:
                repo.create_file(path_in_repo, f"Add KML {kml_id}", content, branch="main")

            print(f"✅ Uploaded {path_in_repo} to GitHub.")
            return f"https://krishnaSureshFor.github.io/tnforest_kml_to_grid_v2.0/public_kml/{kml_id}.kml"

        except Exception as e:
            print("GitHub push failed:", e)
            return None

    class PDF(FPDF):
        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(80, 80, 80)
            self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    pdf = PDF("P", "mm", "A4")
    pdf.set_auto_page_break(auto=True, margin=15)

    # -------------------------------
    # Page 1 — Header + Map
    # -------------------------------
    pdf.add_page()
    if os.path.exists(EMBLEM_PATH):
        pdf.image(EMBLEM_PATH, x=93, y=8, w=25)
    pdf.set_y(35)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "FOREST DEPARTMENT", ln=1, align="C")
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, title_text, ln=1, align="C")

    # Map image
    tmp_dir = tempfile.gettempdir()
    unique_id = uuid.uuid4().hex
    map_img = os.path.join(tmp_dir, f"map_overlay_{unique_id}.png")
    fig, ax = plt.subplots(figsize=(7, 5.8))
    merged_gdf = gpd.GeoSeries([merged_ll], crs="EPSG:4326").to_crs(3857)
    grid_gdf = gpd.GeoSeries(cells_ll, crs="EPSG:4326").to_crs(3857)
    merged_gdf.boundary.plot(ax=ax, color="red", linewidth=3)
    grid_gdf.boundary.plot(ax=ax, color="red", linewidth=1)
    if overlay_gdf is not None and not overlay_gdf.empty:
        overlay_gdf.to_crs(3857).boundary.plot(ax=ax, color="#FFD700", linewidth=3)
    ctx.add_basemap(ax, crs=3857, source=ctx.providers.Esri.WorldImagery, attribution=False)
    ax.axis("off")
    plt.tight_layout(pad=0.1)
    fig.savefig(map_img, dpi=250, bbox_inches="tight")
    plt.close(fig)
    pdf.image(map_img, x=MAP_X, y=MAP_Y, w=MAP_W, h=MAP_H)

    # Legend box
    legend_y = MAP_Y + MAP_H + LEGEND_GAP
    pdf.set_y(legend_y)
    pdf.set_fill_color(245, 245, 240)
    pdf.set_draw_color(180, 180, 180)
    pdf.rect(MAP_X, legend_y, MAP_W, 40, style="FD")
    pdf.set_font("Helvetica", "", 11)
    col1 = [
        f"Range: {user_inputs['range_name']}",
        f"RF: {user_inputs['rf_name']}",
        f"Beat: {user_inputs['beat_name']}",
        f"Year: {user_inputs['year_of_work']}",
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

    # -------------------------------
    # Page 2 — GPS Table (if overlay)
    # -------------------------------
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
                pdf.cell(25, 7, str(row), 1)
                pdf.cell(75, 7, f"{lat:.6f}", 1, align="R")
                pdf.cell(75, 7, f"{lon:.6f}", 1, align="R")
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


    # -------------------------------
    # Invasive Grid Map Box (same size as Page 1 map) — "Invasive Grid Fall"
    # Place below GPS table if space, otherwise create new page.
    # -------------------------------
    if overlay_gdf is not None and not overlay_gdf.empty:
        try:
            # compute overlay accounting df
            df_overlay, overlay_area_ha, total_grid_area_ha = compute_overlay_area_by_grid(cells_ll, overlay_gdf)

            # Prepare clipped grid geometries (cell ∩ overlay)
            from shapely.ops import unary_union
            overlay_union = unary_union(overlay_gdf.geometry)
            clipped_rows = []
            for i, cell in enumerate(cells_ll, start=1):
                inter = cell.intersection(overlay_union)
                if inter.is_empty:
                    continue
                clipped_rows.append({"grid_id": i, "geometry": inter})

            if clipped_rows:
                clipped_gdf = gpd.GeoDataFrame(clipped_rows, geometry=[r['geometry'] for r in clipped_rows], crs="EPSG:4326")
                # compute label points using representative_point
                clipped_gdf['label_pt'] = clipped_gdf.geometry.representative_point()

                # Plot map (same visual layout as Page 1)
                tmp_dir = tempfile.gettempdir()
                invasive_map_img = os.path.join(tmp_dir, f"map_invasive_{unique_id}.png")
                fig, ax = plt.subplots(figsize=(7, 5.8))
                # reproject to web mercator for basemap plotting
                clipped_3857 = clipped_gdf.to_crs(3857)
                overlay_3857 = overlay_gdf.to_crs(3857)
                # Plot clipped grid fill lightly and boundary
                clipped_3857.plot(ax=ax, edgecolor='red', facecolor='none', linewidth=1)
                # Plot overlay boundary on top
                overlay_3857.boundary.plot(ax=ax, color="#FFD700", linewidth=3)
                # Add basemap (Esri World Imagery)
                ctx.add_basemap(ax, crs=3857, source=ctx.providers.Esri.WorldImagery, attribution=False)
                ax.axis('off')
                # Add labels at representative points (project to 3857)
                for idx, row in clipped_3857.iterrows():
                    try:
                        pt = row['geometry'].representative_point()
                        # representative_point is in 3857 already for clipped_3857
                        ax.text(pt.x, pt.y, str(int(clipped_gdf.iloc[idx]['grid_id'])), fontsize=8, ha='center', va='center', color='#03fcfc')
                    except Exception:
                        pass

                # Add corner point labels matching the GPS table S.No
                pt_idx = 1
                for geom in overlay_3857.geometry:
                    if geom.is_empty:
                        continue
                    coords = []
                    if geom.geom_type == "Polygon":
                        coords = list(geom.exterior.coords)
                    elif geom.geom_type == "MultiPolygon":
                        for part in geom.geoms:
                            coords.extend(list(part.exterior.coords))
                    for x, y, *_ in coords:
                        ax.text(x, y, str(pt_idx), fontsize=8, ha='center', va='center', color='black', weight='bold', bbox=dict(facecolor='white', alpha=0.7, pad=0.5, edgecolor='none'))
                        pt_idx += 1

                plt.tight_layout(pad=0.1)
                fig.savefig(invasive_map_img, dpi=250, bbox_inches='tight')
                plt.close(fig)

                # Decide whether to place on same page
                y_now = pdf.get_y()
                page_bottom_limit = 297 - 15  # A4 height minus bottom margin in mm
                # If placing map would overflow, create new page
                if y_now + MAP_H + 20 > page_bottom_limit:
                    pdf.add_page()
                # place title for map
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 8, "Invasive Grid Fall", ln=1, align="C")
                pdf.ln(2)
                # place image at same coords as page 1 map (MAP_X, MAP_Y, MAP_W, MAP_H)
                pdf.image(invasive_map_img, x=MAP_X, y=pdf.get_y(), w=MAP_W, h=MAP_H)
                # move cursor below the image
                pdf.set_y(pdf.get_y() + MAP_H + 4)
            else:
                # No intersecting clipped grids — just note it
                pdf.set_font("Helvetica", "I", 10)
                pdf.cell(0, 8, "No grid cells intersect the overlay — map not generated.", ln=1)
        except Exception as _e:
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 8, f"Invasive map generation failed: {_e}", ln=1)

    # -------------------------------
    # Now add the second table (Grid ID | Area inside overlay) below the map (new page if needed)
    # -------------------------------
    if overlay_gdf is not None and not overlay_gdf.empty:
        try:
            # Ensure df_overlay exists
            if 'df_overlay' not in locals():
                df_overlay, overlay_area_ha, total_grid_area_ha = compute_overlay_area_by_grid(cells_ll, overlay_gdf)

            # If table won't fit on remaining space, start a new page
            rows_needed = len(df_overlay) if df_overlay is not None else 0
            est_height = rows_needed * 8 + 40  # rough estimate in mm (8mm per row)
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
                    # set colored grid id
                    pdf.set_text_color(0, 0, 0)
                    pdf.cell(30, 8, str(int(row["grid_id"])), 1, align="C")
                    pdf.cell(80, 8, f"{row['intersection_area_ha']:.4f}", 1, align="R")
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

            # Total area left-aligned at bottom (rounded up)
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, f"TOTAL AREA INSIDE OVERLAY: {math.ceil(total_grid_area_ha)} Ha", ln=1, align="L")
        except Exception as _e:
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 8, f"Overlay detail table generation failed: {_e}", ln=1)


    # -------------------------------
    # QR Code + Auto-uploaded KML
    # -------------------------------
    y_pos = pdf.get_y()
    try:
        if labeled_kml:
            kml_id = str(uuid.uuid4())[:8]
            tmp_dir = tempfile.gettempdir()
            kml_path = os.path.join(tmp_dir, f"{kml_id}.kml")
            with open(kml_path, "w", encoding="utf-8") as f:
                f.write(labeled_kml)

            # ✅ Push to GitHub
            live_kml_url = push_kml_to_repo(kml_path, kml_id)
            if not live_kml_url:
                live_kml_url = f"https://krishnaSureshFor.github.io/tnforest_kml_to_grid_v2.0/viewer/?id={kml_id}"
            else:
                live_kml_url = f"https://krishnaSureshFor.github.io/tnforest_kml_to_grid_v2.0/viewer/?id={kml_id}"

        else:
            live_kml_url = "https://krishnaSureshFor.github.io/tnforest_kml_to_grid_v2.0"

        # Generate QR
        qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H)
        qr.add_data(live_kml_url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

        img = img.resize((450, 450), Image.NEAREST)
        buf = BytesIO()
        img.save(buf, format="PNG", compress_level=0, dpi=(300, 300))
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

    except Exception as e:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(255, 0, 0)
        pdf.cell(0, 10, f"QR generation failed: {e}", ln=1, align="C")
        pdf.set_text_color(0, 0, 0)

    # Cleanup temp images
    try:
        if 'map_img' in locals() and os.path.exists(map_img):
            os.remove(map_img)
        if 'invasive_map_img' in locals() and os.path.exists(invasive_map_img):
            os.remove(invasive_map_img)
    except Exception:
        pass

    result = pdf.output(dest="S")
    return bytes(result) if isinstance(result, (bytes, bytearray)) else result.encode("latin1", errors="ignore")

# ================================================================
# MAIN APP CONTROL FLOW — Runs only on Generate
# ================================================================

# 1️⃣ Collect user input values into session state on every render
st.session_state["user_inputs"] = {
    "range_name": st.session_state.get("range_name", range_name),
    "rf_name": st.session_state.get("rf_name", rf_name),
    "beat_name": st.session_state.get("beat_name", beat_name),
    "year_of_work": st.session_state.get("year_of_work", year_of_work),
}

# Read widget state directly (no reassignment)
title_text = st.session_state.get("title_text", title_text)
density = st.session_state.get("density", density)
area_invasive = st.session_state.get("area_invasive", area_invasive)
cell_size = st.session_state.get("cell_size", cell_size)

# ================================================================
# CACHED OUTPUT GENERATOR
# ================================================================
@st.cache_data(show_spinner=False)
def generate_all_outputs(aoi_path, overlay_path, user_inputs, cell_size, title_text, density, area_invasive):
    # --- Read and clean AOI to polygons only ---
    gdf = read_kml_safely(aoi_path)
    gdf = clean_polygon_gdf(gdf)
    if gdf is None or gdf.empty:
        raise ValueError("AOI file has no valid polygon geometries. Remove points/lines/empty Placemarks and try again.")

    polygons = gdf.geometry
    cells_ll, merged_ll = make_grid_exact_clipped(polygons, cell_size)

    overlay_gdf = None
    if overlay_path:
        overlay_gdf = read_kml_safely(overlay_path).to_crs(4326)
        overlay_gdf = clean_polygon_gdf(overlay_gdf)

    grid_only_kml = generate_grid_only_kml(cells_ll, merged_ll, user_inputs)
    labeled_kml = generate_labeled_kml(cells_ll, merged_ll, user_inputs, overlay_gdf)

    pdf_bytes = build_pdf_report_standard(
        cells_ll, merged_ll, user_inputs, cell_size,
        overlay_gdf, title_text, density, area_invasive,
        labeled_kml=labeled_kml  # ✅ new argument added here
    )

    return {
        "grid_only_kml": grid_only_kml,
        "labeled_kml": labeled_kml,
        "pdf_bytes": pdf_bytes,
        "overlay_gdf": overlay_gdf,
        "cells_ll": cells_ll,
        "merged_ll": merged_ll,
    }

# 2️⃣ Only execute heavy logic if user pressed Generate
if generate_click:
    st.session_state["generated"] = True

# 3️⃣ Only display map + outputs once generated

# --- Edit AOI Page ---
if st.session_state.get("edit_mode", False):
    st.title("Edit AOI Boundary")
    aoi_gdf = st.session_state.get("aoi_gdf")
    if aoi_gdf is None:
        st.warning("No AOI loaded to edit")
        st.stop()

    st.write("Edit the AOI boundary. You can draw or upload an edited KML. After editing, click Re-run Grid Formation.")

    if LEAFMAP_AVAILABLE:
        try:
            m = leafmap.Map(center=[aoi_gdf.geometry.centroid.y.mean(), aoi_gdf.geometry.centroid.x.mean()], zoom=13)
            m.add_gdf(aoi_gdf, layer_name="AOI")
            m.add_draw_control(draw_options={"polyline": False, "polygon": True, "circle": False, "rectangle": True, "marker": False}, edit_options={"poly": {"allowIntersection": False}})
            m.to_streamlit(height=600)
        except Exception:
            st.info("Leafmap failed to initialize; please upload edited KML instead.")

    edited_file = st.file_uploader("Upload Edited AOI KML/KMZ (or leave empty if you edited on map)", type=["kml", "kmz"])
    if edited_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".kml") as tmp:
                tmp.write(edited_file.read())
                tpath = tmp.name
            edited_gdf = read_kml_safely(tpath)
            st.session_state["aoi_gdf_edited_temp"] = edited_gdf
            st.success("Edited AOI loaded (temporary). Click 'Re-upload Edited AOI' to apply it.")
        except Exception as e:
            st.error(f"Failed to read edited file: {e}")

    if st.button("Re-upload Edited AOI"):
        if st.session_state.get("aoi_gdf_edited_temp") is not None:
            st.session_state["aoi_gdf"] = st.session_state.pop("aoi_gdf_edited_temp")
            st.session_state["generated"] = False
            st.session_state["edit_mode"] = False
            st.experimental_rerun()
        else:
            st.warning("No edited AOI available. Upload or edit on the map first.")

    # Download modified AOI
    try:
        tmpd = tempfile.mkdtemp()
        modified_kml = os.path.join(tmpd, "Modified_AOI.kml")
        st.session_state["aoi_gdf"].to_file(modified_kml, driver="KML")
        with open(modified_kml, "rb") as f:
            # if styled_download_button exists in your app, it will use; otherwise fallback to streamlit button shown elsewhere
            try:
                styled_download_button("Download Modified KML", f.read(), "Modified_AOI.kml", "application/vnd.google-earth.kml+xml", icon="📥", bg="#0ea5e9")
            except Exception:
                st.download_button("Download Modified KML", f.read(), file_name="Modified_AOI.kml", mime="application/vnd.google-earth.kml+xml")
    except Exception:
        pass

    st.stop()
if st.session_state.get("generated", False):

    st.success("✅ Grid successfully generated! Scroll below to preview map and downloads.")
    aoi_path, ov_path = None, None

    # Handle AOI (required)
    if uploaded_aoi:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".kmz" if uploaded_aoi.name.endswith(".kmz") else ".kml") as tmp:
            tmp.write(uploaded_aoi.read())
            aoi_path = tmp.name
    else:
        st.warning("⚠️ Please upload an AOI file before generating.")
        st.stop()

    # Handle Overlay (optional)
    if overlay_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".kmz" if overlay_file.name.endswith(".kmz") else ".kml") as tmp:
            tmp.write(overlay_file.read())
            ov_path = tmp.name
    else:
        ov_path = None

    # ============================================================
    # Run cached generator (no recomputation, no reload on download)
    # ============================================================
    try:
        outputs = generate_all_outputs(
            aoi_path, ov_path,
            st.session_state["user_inputs"],
            cell_size, title_text, density, area_invasive
        )
    except ValueError as e:
        st.error(f"❌ {e}")
        st.stop()

    for k, v in outputs.items():
        st.session_state[k] = v

    # ============================================================
    # MAP PREVIEW — Static and stable
    # ============================================================
    m = folium.Map(location=[11, 78.5], zoom_start=8)

    gdf_for_bounds = read_kml_safely(aoi_path)
    gdf_for_bounds = clean_polygon_gdf(gdf_for_bounds)
    if gdf_for_bounds is None or gdf_for_bounds.empty:
        st.error("AOI file has no valid polygon geometries for map preview.")
        st.stop()

    aoi_union = unary_union(gdf_for_bounds.geometry)

    # AOI boundary
    folium.GeoJson(
        mapping(aoi_union),
        style_function=lambda x: {"color": "red", "weight": 3, "fillOpacity": 0}
    ).add_to(m)

    # Grid cells
    for c in st.session_state["cells_ll"]:
        folium.GeoJson(
            mapping(c),
            style_function=lambda x: {"color": "red", "weight": 1, "fillOpacity": 0}
        ).add_to(m)

    # Overlay
    if st.session_state["overlay_gdf"] is not None and not st.session_state["overlay_gdf"].empty:
        for g in st.session_state["overlay_gdf"].geometry:
            if g.is_empty:
                continue
            folium.GeoJson(
                mapping(g),
                style_function=lambda x: {"color": "#FFD700", "weight": 3, "fillOpacity": 0}
            ).add_to(m)

    # Fit bounds and display
    bounds = [
        [aoi_union.bounds[1], aoi_union.bounds[0]],
        [aoi_union.bounds[3], aoi_union.bounds[2]],
    ]
    m.fit_bounds(bounds)
    st_folium(m, width=1200, height=700)

    # ============================================================
    # DOWNLOADS — No reload on click
    # ============================================================
    st.markdown("### 💾 Downloads")

    # Single Shapefile zip (contains two folders: Grid_with_ID and Grid_Only)
    def export_shapefiles_combined(cells_ll, overlay_gdf=None):
        """
        Create a ZIP bytes containing two folders:
         - Grid_with_ID (ESRI shapefile set with grid_id attribute)
         - Grid_Only (ESRI shapefile set without attributes)
        Returns bytes of the ZIP file.
        """
        root_tmp = tempfile.mkdtemp()
        try:
            # prepare GeoDataFrames
            gdf_grid_id = gpd.GeoDataFrame({"grid_id": range(1, len(cells_ll) + 1)}, geometry=cells_ll, crs="EPSG:4326")
            gdf_grid_only = gpd.GeoDataFrame(geometry=cells_ll, crs="EPSG:4326")

            # Folder1
            folder1 = os.path.join(root_tmp, "Grid_with_ID")
            os.makedirs(folder1, exist_ok=True)
            # Fiona/GeoPandas require file path prefix without extension
            out1 = os.path.join(folder1, "Grid_with_ID")
            gdf_grid_id.to_file(out1 + ".shp", driver="ESRI Shapefile")

            # Folder2
            folder2 = os.path.join(root_tmp, "Grid_Only")
            os.makedirs(folder2, exist_ok=True)
            out2 = os.path.join(folder2, "Grid_Only")
            gdf_grid_only.to_file(out2 + ".shp", driver="ESRI Shapefile")

            # If overlay_gdf is provided, also include it as overlay.shp in Grid_with_ID folder (optional)
            if overlay_gdf is not None and not overlay_gdf.empty:
                try:
                    overlay_out = os.path.join(folder1, "Overlay")
                    overlay_gdf.to_file(overlay_out + ".shp", driver="ESRI Shapefile")
                except Exception:
                    # ignore overlay write failure, but continue packaging
                    pass

            # Zip the whole root_tmp into bytes
            zip_buf = BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as z:
                for foldername, subfolders, filenames in os.walk(root_tmp):
                    for filename in filenames:
                        file_path = os.path.join(foldername, filename)
                        # archive name: remove root_tmp prefix and keep folder structure
                        arcname = os.path.relpath(file_path, root_tmp)
                        z.write(file_path, arcname=arcname)
            zip_buf.seek(0)
            return zip_buf.getvalue()
        finally:
            # cleanup disk files
            try:
                shutil.rmtree(root_tmp)
            except Exception:
                pass

    try:
        shp_bytes = export_shapefiles_combined(st.session_state["cells_ll"], st.session_state.get("overlay_gdf"))
        st.download_button(
            "📦 Download Shapefiles (Grid+Overlay & Grid Only)",
            shp_bytes,
            file_name="shapefiles_grid.zip",
            mime="application/zip",
        )
    except Exception as e:
        st.error(f"Shapefile export failed: {e}")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button(
            "📦 Download Grid Only KML",
            st.session_state["grid_only_kml"],
            file_name="grid_only.kml",
            mime="application/vnd.google-earth.kml+xml",
        )
    with c2:
        st.download_button(
            "🧾 Download Labeled + Overlay KML",
            st.session_state["labeled_kml"],
            file_name="merged_labeled.kml",
            mime="application/vnd.google-earth.kml+xml",
        )
    with c3:
        if generate_pdf:
            st.download_button(
                "📄 Download Invasive Report (PDF)",
                st.session_state["pdf_bytes"],
                file_name="Invasive_Report.pdf",
                mime="application/pdf",
            )

else:
    st.info("👆 Use Sidebar Arrow to Upload AOI (KML/KMZ) and Overlay, adjust details, then click ▶ **Generate Grid**.")

# Optional: Hide Streamlit spinner for smoother UI
st.markdown("<style>.stSpinner{display:none}</style>", unsafe_allow_html=True)

# Clean up temporary uploaded files to prevent disk leaks
try:
    if 'aoi_path' in locals() and aoi_path and os.path.exists(aoi_path):
        os.remove(aoi_path)
    if 'ov_path' in locals() and ov_path and os.path.exists(ov_path):
        os.remove(ov_path)
except Exception:
    pass
