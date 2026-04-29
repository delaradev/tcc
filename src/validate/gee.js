var MUNICIPALITIES = ee.FeatureCollection(
	"projects/elr-ifrs-tcc/assets/RS_Municipios_2025",
);
var TARGET_AREA = MUNICIPALITIES.filter(
	ee.Filter.eq("NM_MUN", "Salto do Jacuí"),
);
Map.centerObject(TARGET_AREA, 11);
Map.addLayer(
	TARGET_AREA,
	{ color: "#ff0000", opacity: 0.5 },
	"Study Area Boundary",
);

var ACQUISITION_CONFIG = {
	START_DATE: "2023-01-01",
	END_DATE: "2023-12-31",
	MAX_CLOUD_COVER: 50,
	SCALE_FACTOR: 0.0000275,
	OFFSET: -0.2,
};
var FALLBACK_CONFIG = {
	START_DATE: "2022-01-01",
	END_DATE: "2023-12-31",
	MAX_CLOUD_COVER: 80,
};

function getLandsatCollection() {
	var missions = [
		ee.ImageCollection("LANDSAT/LT05/C02/T1_L2"),
		ee.ImageCollection("LANDSAT/LE07/C02/T1_L2"),
		ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"),
		ee.ImageCollection("LANDSAT/LC09/C02/T1_L2"),
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
		}),
	);
}

var landsatCollection = getLandsatCollection();
var imageCount = landsatCollection.size().getInfo();
if (imageCount === 0) {
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
		fallbackCollections[0].merge(fallbackCollections[1]),
	);
	imageCount = landsatCollection.size().getInfo();
	print("Fallback active. Images available: " + imageCount);
}

function addSpectralIndices(image) {
	var optical = image
		.select("SR_B.")
		.multiply(ACQUISITION_CONFIG.SCALE_FACTOR)
		.add(ACQUISITION_CONFIG.OFFSET);
	var bands = {
		blue: optical.select("SR_B1"),
		green: optical.select("SR_B2"),
		red: optical.select("SR_B3"),
		nir: optical.select("SR_B4"),
		swir2: optical.select("SR_B7"),
	};
	var evi = bands.nir
		.expression("2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))", {
			NIR: bands.nir,
			RED: bands.red,
			BLUE: bands.blue,
		})
		.rename("EVI");
	var bsi = bands.swir2
		.expression(
			"((SWIR2 + RED) - (NIR + BLUE)) / ((SWIR2 + RED) + (NIR + BLUE))",
			{
				SWIR2: bands.swir2,
				RED: bands.red,
				NIR: bands.nir,
				BLUE: bands.blue,
			},
		)
		.rename("BSI");
	var green = bands.green.rename("GREEN");
	return image.addBands([evi, bsi, green]);
}

var processedCollection = landsatCollection.map(addSpectralIndices);
var annualComposite = ee.Image.cat([
	processedCollection.select("EVI").max().rename("EVI_max"),
	processedCollection.select("BSI").max().rename("BSI_max"),
	processedCollection.select("GREEN").median().rename("GREEN_median"),
]).clip(TARGET_AREA);

var VISUALIZATION_PARAMS = {
	bands: ["EVI_max", "BSI_max", "GREEN_median"],
	min: [0.0, -0.1, 0.0],
	max: [0.7, 0.3, 0.15],
	gamma: 1.1,
};
Map.addLayer(annualComposite, VISUALIZATION_PARAMS, "Annual CPIC Composite");

var EXPORT_CONFIG = {
	scale: 30,
	maxPixels: 1e13,
	fileFormat: "GeoTIFF",
	folder: "GEE_EXPORT",
};
Export.image.toDrive({
	image: annualComposite,
	description: "Landsat_CPIC_Composite_2023",
	fileNamePrefix: "landsat_cpic_2023",
	region: TARGET_AREA.geometry(),
	scale: EXPORT_CONFIG.scale,
	maxPixels: EXPORT_CONFIG.maxPixels,
	fileFormat: EXPORT_CONFIG.fileFormat,
	folder: EXPORT_CONFIG.folder,
});
