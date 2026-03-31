// ============================================================
// GERAÇÃO DE COMPOSIÇÕES LANDSAT PARA PIVÔS CENTRAIS
// Baseado estritamente em: Liu et al. (2023) - ISPRS Journal
// Composição: [EVI_max, BSI_max, GREEN_median] - Landsat 30m
// ============================================================

// =========================
// 1. CARREGAR SHAPEFILE IBGE
// =========================
var municipios = ee.FeatureCollection(
	"projects/elr-ifrs-tcc/assets/RS_Municipios_2025",
);
var salto = municipios.filter(ee.Filter.eq("NM_MUN", "Salto do Jacuí"));

Map.centerObject(salto, 11);
Map.addLayer(salto, { color: "red" }, "Salto do Jacuí");

// =========================
// 2. COLETAR LANDSAT (TODAS AS MISSÕES)
// =========================
var landsat5 = ee
	.ImageCollection("LANDSAT/LT05/C02/T1_L2")
	.filterBounds(salto)
	.filterDate("2023-01-01", "2023-12-31")
	.filter(ee.Filter.lt("CLOUD_COVER", 50));

var landsat7 = ee
	.ImageCollection("LANDSAT/LE07/C02/T1_L2")
	.filterBounds(salto)
	.filterDate("2023-01-01", "2023-12-31")
	.filter(ee.Filter.lt("CLOUD_COVER", 50));

var landsat8 = ee
	.ImageCollection("LANDSAT/LC08/C02/T1_L2")
	.filterBounds(salto)
	.filterDate("2023-01-01", "2023-12-31")
	.filter(ee.Filter.lt("CLOUD_COVER", 50));

var landsat9 = ee
	.ImageCollection("LANDSAT/LC09/C02/T1_L2")
	.filterBounds(salto)
	.filterDate("2023-01-01", "2023-12-31")
	.filter(ee.Filter.lt("CLOUD_COVER", 50));

var allLandsat = landsat5.merge(landsat7).merge(landsat8).merge(landsat9);
print("Número de imagens Landsat:", allLandsat.size());

// Verificar se há imagens
if (allLandsat.size().getInfo() === 0) {
	print("NENHUMA IMAGEM LANDSAT ENCONTRADA!");
	print("Tentando expandir o período ou reduzir filtro de nuvens...");

	// Fallback: período mais amplo e mais tolerância para nuvens
	var landsat5_fallback = ee
		.ImageCollection("LANDSAT/LT05/C02/T1_L2")
		.filterBounds(salto)
		.filterDate("2022-01-01", "2023-12-31")
		.filter(ee.Filter.lt("CLOUD_COVER", 80));

	var landsat8_fallback = ee
		.ImageCollection("LANDSAT/LC08/C02/T1_L2")
		.filterBounds(salto)
		.filterDate("2022-01-01", "2023-12-31")
		.filter(ee.Filter.lt("CLOUD_COVER", 80));

	allLandsat = landsat5_fallback.merge(landsat8_fallback);
	print("Número de imagens (fallback):", allLandsat.size());
}

// =========================
// 3. FUNÇÃO PARA CALCULAR ÍNDICES (CONFORME ARTIGO)
// =========================
function addIndicesLandsat(img) {
	// Aplicar escala e offset (Landsat Collection 2)
	var opticalBands = img.select("SR_B.").multiply(0.0000275).add(-0.2);

	// Selecionar bandas (mapeamento Landsat)
	var blue = opticalBands.select("SR_B1"); // 450-510 nm
	var green = opticalBands.select("SR_B2"); // 530-590 nm
	var red = opticalBands.select("SR_B3"); // 640-670 nm
	var nir = opticalBands.select("SR_B4"); // 850-880 nm
	var swir1 = opticalBands.select("SR_B5"); // 1570-1650 nm
	var swir2 = opticalBands.select("SR_B7"); // 2110-2290 nm

	// EVI: 2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))
	var evi = nir
		.expression("2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))", {
			NIR: nir,
			RED: red,
			BLUE: blue,
		})
		.rename("EVI");

	// BSI: ((SWIR2 + RED) - (NIR + BLUE)) / ((SWIR2 + RED) + (NIR + BLUE))
	var bsi = swir2
		.expression(
			"((SWIR2 + RED) - (NIR + BLUE)) / ((SWIR2 + RED) + (NIR + BLUE))",
			{
				SWIR2: swir2,
				RED: red,
				NIR: nir,
				BLUE: blue,
			},
		)
		.rename("BSI");

	// Banda verde (normalizada)
	var greenNorm = green.rename("GREEN");

	return img.addBands([evi, bsi, greenNorm]);
}

// Aplicar índices
var landsatWithIndices = allLandsat.map(addIndicesLandsat);

// =========================
// 4. COMPOSIÇÃO ANUAL (CONFORME ARTIGO)
// =========================
// EVI_max: Máximo anual do EVI
var eviMax = landsatWithIndices.select("EVI").max().rename("EVI_max").toFloat();

// BSI_max: Máximo anual do BSI
var bsiMax = landsatWithIndices.select("BSI").max().rename("BSI_max").toFloat();

// GREEN_median: Mediana anual da banda verde
var greenMedian = landsatWithIndices
	.select("GREEN")
	.median()
	.rename("GREEN_median")
	.toFloat();

// =========================
// 5. STACK FINAL (3 BANDAS - EXATAMENTE COMO O ARTIGO)
// =========================
var stack = ee.Image.cat([eviMax, bsiMax, greenMedian])
	.clip(salto)
	.rename(["EVI_max", "BSI_max", "GREEN_median"]);

// =========================
// 6. ESTATÍSTICAS PARA VISUALIZAÇÃO
// =========================
// Calcular percentis para melhor visualização
var percentiles = stack.reduceRegion({
	reducer: ee.Reducer.percentile([2, 50, 98]),
	geometry: salto,
	scale: 30,
	maxPixels: 1e9,
	bestEffort: true,
});

print("Percentis para visualização:", {
	EVI: {
		min: percentiles.get("EVI_max_p2"),
		median: percentiles.get("EVI_max_p50"),
		max: percentiles.get("EVI_max_p98"),
	},
	BSI: {
		min: percentiles.get("BSI_max_p2"),
		median: percentiles.get("BSI_max_p50"),
		max: percentiles.get("BSI_max_p98"),
	},
	GREEN: {
		min: percentiles.get("GREEN_median_p2"),
		median: percentiles.get("GREEN_median_p50"),
		max: percentiles.get("GREEN_median_p98"),
	},
});

// =========================
// 7. VISUALIZAÇÃO NO MAPA
// =========================
var visParams = {
	bands: ["EVI_max", "BSI_max", "GREEN_median"],
	min: [0, -0.1, 0],
	max: [0.7, 0.3, 0.15],
	gamma: 1.1,
};

Map.addLayer(stack, visParams, "Stack Landsat - Padrão Artigo");

// =========================
// 8. EXPORTAÇÃO (APENAS GEOTIFF - FORMATO CORRETO)
// =========================
print("\nINICIANDO EXPORTAÇÃO...");
print("   Formato: GeoTIFF");
print("   Resolução: 30 metros");
print("   Bandas: EVI_max, BSI_max, GREEN_median");
print("   Região: Salto do Jacuí");
print("");

// Exportação principal em GeoTIFF
Export.image.toDrive({
	image: stack,
	description: "Landsat_CPIC_Stack_2023",
	folder: "GEE_EXPORT",
	fileNamePrefix: "landsat_cpic_2023",
	region: salto.geometry(),
	scale: 30,
	maxPixels: 1e13,
	fileFormat: "GeoTIFF",
});

// =========================
// 9. EXPORTAÇÃO DE VISUALIZAÇÃO (OPCIONAL)
// =========================
// Para visualização, criar uma imagem RGB falsa
var rgbVis = stack.visualize(visParams);

// Exportar visualização como GeoTIFF também (não PNG)
Export.image.toDrive({
	image: rgbVis,
	description: "Landsat_CPIC_Visual_2023",
	folder: "GEE_EXPORT",
	fileNamePrefix: "landsat_cpic_visual_2023",
	region: salto.geometry(),
	scale: 30,
	maxPixels: 1e13,
	fileFormat: "GeoTIFF",
});

// =========================
// 10. INFORMAÇÕES ADICIONAIS
// =========================
print("\nINFORMAÇÕES DA IMAGEM:");
print(
	"  Dimensões aproximadas:",
	stack.select("EVI_max").projection().nominalScale(),
);
print("  CRS:", stack.select("EVI_max").projection());
print("  Área aproximada (ha):", salto.geometry().area().divide(10000));

// =========================
// 11. VERIFICAR QUALIDADE DOS DADOS
// =========================
// Contar pixels válidos por banda
var validPixels = stack.reduceRegion({
	reducer: ee.Reducer.count(),
	geometry: salto,
	scale: 30,
	maxPixels: 1e9,
	bestEffort: true,
});

print("Pixels válidos por banda:", {
	EVI_max: validPixels.get("EVI_max"),
	BSI_max: validPixels.get("BSI_max"),
	GREEN_median: validPixels.get("GREEN_median"),
});
