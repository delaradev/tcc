/**
 * ============================================================================
 * LANDSAT-BASED CENTER PIVOT IRRIGATION CROPLAND (CPIC) COMPOSITION
 * ============================================================================
 *
 * Implements the annual composite generation methodology described in:
 * Liu et al. (2023) - ISPRS Journal of Photogrammetry and Remote Sensing
 *
 * Output: 3-band GeoTIFF [EVI_max, BSI_max, GREEN_median] at 30m resolution
 *
 * @author Emanuel de Lara Ruas
 * @version 1.0.0
 *
 * ============================================================================
 */

// ----------------------------------------------------------------------------
// 1. GEOGRAPHIC SCOPE DEFINITION
// ----------------------------------------------------------------------------

/**
 * Load municipality boundaries from GEE Assets
 * @type {ee.FeatureCollection}
 */
var MUNICIPALITIES = ee.FeatureCollection(
	"projects/elr-ifrs-tcc/assets/RS_Municipios_2025"
);

/**
 * Target municipality geometry
 * @type {ee.FeatureCollection}
 */
var TARGET_AREA = MUNICIPALITIES.filter(
	ee.Filter.eq("NM_MUN", "Salto do Jacuí")
);

// Configure map viewport and visualization
Map.centerObject(TARGET_AREA, 11);
Map.addLayer(
	TARGET_AREA,
	{ color: "#ff0000", opacity: 0.5 },
	"Study Area Boundary"
);

// ----------------------------------------------------------------------------
// 2. LANDSAT DATA ACQUISITION
// ----------------------------------------------------------------------------

/**
 * Collection parameters aligned with Liu et al. (2023)
 * @constant
 */
var ACQUISITION_CONFIG = {
	START_DATE: "2023-01-01",
	END_DATE: "2023-12-31",
	MAX_CLOUD_COVER: 50,
	SCALE_FACTOR: 0.0000275,
	OFFSET: -0.2,
};

/**
 * Fallback parameters for cloud-prone regions
 * @constant
 */
var FALLBACK_CONFIG = {
	START_DATE: "2022-01-01",
	END_DATE: "2023-12-31",
	MAX_CLOUD_COVER: 80
};

/**
 * Retrieve Landsat imagery from all operational missions
 * @returns {ee.ImageCollection} Merged collection of Landsat 5-9 imagery
 */
function getLandsatCollection() {
	var missions = [
		ee.ImageCollection("LANDSAT/LT05/C02/T1_L2"), // Landsat 5
		ee.ImageCollection("LANDSAT/LE07/C02/T1_L2"), // Landsat 7
		ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"), // Landsat 8
		ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"), // Landsat 9
	];

	var collections = missions.map(function (mission) {
		return mission
			.filterBounds(TARGET_AREA)
			.filterDate(ACQUISITION_CONFIG.START_DATE, ACQUISITION_CONFIG.END_DATE)
			.filter(ee.Filter.lt("CLOUD_COVER", ACQUISITION_CONFIG.MAX_CLOUD_COVER));
	});

	return ee.ImageCollection(
		collections.reduce(function (acc, col) {
			return acc.merge(col);
		})
	);
}

var landsatCollection = getLandsatCollection();
var imageCount = landsatCollection.size().getInfo();

// Fallback strategy for data-scarce regions
if (imageCount === 0) {
	console.warn(
		"Insufficient imagery with standard parameters. Activating fallback..."
	);

	var fallbackCollections = [
		ee
			.ImageCollection("LANDSAT/LT05/C02/T1_L2")
			.filterBounds(TARGET_AREA)
			.filterDate(FALLBACK_CONFIG.START_DATE, FALLBACK_CONFIG.END_DATE)
			.filter(ee.Filter.lt("CLOUD_COVER", FALLBACK_CONFIG.MAX_CLOUD_COVER)),
		ee
			.ImageCollection("LANDSAT/LC08/C02/T1_L2")
			.filterBounds(TARGET_AREA)
			.filterDate(FALLBACK_CONFIG.START_DATE, FALLBACK_CONFIG.END_DATE)
			.filter(ee.Filter.lt("CLOUD_COVER", FALLBACK_CONFIG.MAX_CLOUD_COVER)),
	];

	landsatCollection = ee.ImageCollection(
		fallbackCollections[0].merge(fallbackCollections[1])
	);
	imageCount = landsatCollection.size().getInfo();

	console.info("Fallback active. Images available: " + imageCount);
}

print("Landsat imagery summary:", {
	total_images: imageCount,
	temporal_range:
		ACQUISITION_CONFIG.START_DATE + " - " + ACQUISITION_CONFIG.END_DATE,
	max_cloud_cover: ACQUISITION_CONFIG.MAX_CLOUD_COVER + "%"
});

// ----------------------------------------------------------------------------
// 3. SPECTRAL INDEX COMPUTATION
// ----------------------------------------------------------------------------

/**
 * Compute EVI (Enhanced Vegetation Index) and BSI (Bare Soil Index)
 * following the methodology from Liu et al. (2023)
 *
 * @param {ee.Image} image - Raw Landsat surface reflectance image
 * @returns {ee.Image} Image with added EVI, BSI, and GREEN bands
 */
function addSpectralIndices(image) {
	// Apply radiometric calibration (Collection 2 scaling)
	var optical = image
		.select("SR_B.")
		.multiply(ACQUISITION_CONFIG.SCALE_FACTOR)
		.add(ACQUISITION_CONFIG.OFFSET);

	// Spectral band extraction per Landsat specification
	var bands = {
		blue: optical.select("SR_B1"), // 450-510 nm
		green: optical.select("SR_B2"), // 530-590 nm
		red: optical.select("SR_B3"), // 640-670 nm
		nir: optical.select("SR_B4"), // 850-880 nm
		swir2: optical.select("SR_B7"), // 2110-2290 nm
	};

	// Enhanced Vegetation Index (Huete et al., 1997)
	// Optimized for areas with high biomass and atmospheric effects
	var evi = bands.nir
		.expression("2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))", {
			NIR: bands.nir,
			RED: bands.red,
			BLUE: bands.blue
		})
		.rename("EVI");

	// Bare Soil Index (Diek et al., 2017)
	// Captures soil exposure during fallow periods and post-harvest
	var bsi = bands.swir2
		.expression(
			"((SWIR2 + RED) - (NIR + BLUE)) / ((SWIR2 + RED) + (NIR + BLUE))",
			{
				SWIR2: bands.swir2,
				RED: bands.red,
				NIR: bands.nir,
				BLUE: bands.blue,
			}
		)
		.rename("BSI");

	// Raw green band preserved for texture analysis
	var green = bands.green.rename("GREEN");

	return image.addBands([evi, bsi, green]);
}

// Apply spectral indices to the entire collection
var processedCollection = landsatCollection.map(addSpectralIndices);

// ----------------------------------------------------------------------------
// 4. ANNUAL COMPOSITE GENERATION
// ----------------------------------------------------------------------------

/**
 * Generate annual composites following the three-band strategy:
 * - EVI_max: Peak vegetation activity
 * - BSI_max: Maximum soil exposure
 * - GREEN_median: Representative green reflectance
 *
 * This composition mitigates phenological heterogeneity across Brazil's
 * diverse climatic zones (Liu et al., 2023, Section 3.1)
 */

var annualComposite = ee.Image.cat([
	processedCollection.select("EVI").max().rename("EVI_max"),
	processedCollection.select("BSI").max().rename("BSI_max"),
	processedCollection.select("GREEN").median().rename("GREEN_median"),
]).clip(TARGET_AREA);

// ----------------------------------------------------------------------------
// 5. STATISTICAL CHARACTERIZATION
// ----------------------------------------------------------------------------

/**
 * Compute percentile-based statistics for optimal visualization
 * and quality assessment
 */
var STATISTICS_CONFIG = {
	PERCENTILES: [2, 50, 98],
	SCALE_METERS: 30,
	MAX_PIXELS: 1e9,
};

var statistics = annualComposite.reduceRegion({
	reducer: ee.Reducer.percentile(STATISTICS_CONFIG.PERCENTILES),
	geometry: TARGET_AREA,
	scale: STATISTICS_CONFIG.SCALE_METERS,
	maxPixels: STATISTICS_CONFIG.MAX_PIXELS,
	bestEffort: true,
});

/**
 * Extract per-band statistical summaries
 */
var bandStats = {
	evi: {
		min: statistics.get("EVI_max_p2"),
		median: statistics.get("EVI_max_p50"),
		max: statistics.get("EVI_max_p98"),
	},
	bsi: {
		min: statistics.get("BSI_max_p2"),
		median: statistics.get("BSI_max_p50"),
		max: statistics.get("BSI_max_p98"),
	},
	green: {
		min: statistics.get("GREEN_median_p2"),
		median: statistics.get("GREEN_median_p50"),
		max: statistics.get("GREEN_median_p98"),
	},
};

print("Band statistics (2nd-98th percentile):", bandStats);

// ----------------------------------------------------------------------------
// 6. VISUALIZATION CONFIGURATION
// ----------------------------------------------------------------------------

/**
 * Visualization parameters optimized for CPIC identification
 *
 * EVI: 0.0-0.7 (vegetation intensity)
 * BSI: -0.1-0.3 (soil exposure gradient)
 * GREEN: 0.0-0.15 (background texture)
 */
var VISUALIZATION_PARAMS = {
	bands: ["EVI_max", "BSI_max", "GREEN_median"],
	min: [0.0, -0.1, 0.0],
	max: [0.7, 0.3, 0.15],
	gamma: 1.1,
};

Map.addLayer(annualComposite, VISUALIZATION_PARAMS, "Annual CPIC Composite");

// ----------------------------------------------------------------------------
// 7. DATA QUALITY ASSESSMENT
// ----------------------------------------------------------------------------

/**
 * Validate composite completeness by counting valid pixels per band
 */
var validPixelCount = annualComposite.reduceRegion({
	reducer: ee.Reducer.count(),
	geometry: TARGET_AREA,
	scale: STATISTICS_CONFIG.SCALE_METERS,
	maxPixels: STATISTICS_CONFIG.MAX_PIXELS,
	bestEffort: true,
});

print("Quality assessment:", {
	valid_pixels_evi: validPixelCount.get("EVI_max"),
	valid_pixels_bsi: validPixelCount.get("BSI_max"),
	valid_pixels_green: validPixelCount.get("GREEN_median")
});

// ----------------------------------------------------------------------------
// 8. METADATA AND SPATIAL REFERENCE
// ----------------------------------------------------------------------------

/**
 * Extract spatial reference information for documentation
 */
var projection = annualComposite.select("EVI_max").projection();
var nominalScale = projection.nominalScale();
var areaHectares = TARGET_AREA.geometry().area().divide(10000);

print("Spatial metadata:", {
	crs: projection,
	nominal_scale_meters: nominalScale,
	study_area_hectares: areaHectares
});

// ----------------------------------------------------------------------------
// 9. EXPORT CONFIGURATION
// ----------------------------------------------------------------------------

/**
 * Export specifications aligned with downstream processing requirements
 *
 * Format: GeoTIFF with Float32 precision per band
 * Resolution: 30m (original Landsat resolution)
 * Region: Clipped to study area boundary
 */
var EXPORT_CONFIG = {
	scale: 30,
	maxPixels: 1e13,
	fileFormat: "GeoTIFF",
	folder: "GEE_EXPORT"
};

/**
 * Export the core 3-band composite for model training
 */
Export.image.toDrive({
	image: annualComposite,
	description: "Landsat_CPIC_Composite_2023",
	fileNamePrefix: "landsat_cpic_2023",
	region: TARGET_AREA.geometry(),
	scale: EXPORT_CONFIG.scale,
	maxPixels: EXPORT_CONFIG.maxPixels,
	fileFormat: EXPORT_CONFIG.fileFormat,
	folder: EXPORT_CONFIG.folder
});

/**
 * Export a visually-enhanced version for rapid inspection
 */
var visualComposite = annualComposite.visualize(VISUALIZATION_PARAMS);

Export.image.toDrive({
	image: visualComposite,
	description: "Landsat_CPIC_Visualization_2023",
	fileNamePrefix: "landsat_cpic_visual_2023",
	region: TARGET_AREA.geometry(),
	scale: EXPORT_CONFIG.scale,
	maxPixels: EXPORT_CONFIG.maxPixels,
	fileFormat: EXPORT_CONFIG.fileFormat,
	folder: EXPORT_CONFIG.folder
});

// ----------------------------------------------------------------------------
// 10. EXECUTION SUMMARY
// ----------------------------------------------------------------------------

print("=== EXPORT SUMMARY ===");
print("Task 1: landsat_cpic_2023.tif - Raw composite (3-band, Float32)");
print("Task 2: landsat_cpic_visual_2023.tif - RGB visualization");
print("\nTo initiate export:");
print("1. Navigate to the Tasks tab (right panel)");
print('2. Click "Run" for each export task');
print("3. Files will be saved to your Google Drive > GEE_EXPORT folder");
print("\nExpected output:");
print("- Format: GeoTIFF");
print("- Resolution: 30 meters/pixel");
print("- Bands: EVI_max, BSI_max, GREEN_median");
print("- Spatial extent: Salto do Jacuí municipality");

// Export configuration verification
print("\nExport configuration:", {
	geometry_type: TARGET_AREA.geometry().type(),
	export_region: TARGET_AREA.geometry().bounds(),
	export_scale_meters: EXPORT_CONFIG.scale
});
