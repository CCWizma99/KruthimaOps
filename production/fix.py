with open("c:/KruthimaOps/production/app/static/app.js", "r", encoding="utf-8") as f:
    content = f.read()

# Fix the broken getRiskColor
content = content.replace(
    "if (score < 0.75) return Cesium.Color.fromCssColorString('#f97316');\n  return Cesium.Color.fromBytes(r, g, b, Math.round(alpha * 255));\n}",
    "if (score < 0.75) return Cesium.Color.fromCssColorString('#f97316');\n  return Cesium.Color.fromCssColorString('#ef4444');\n}"
)

# Fix the getInterpolatedColor
content = content.replace(
    "return Cesium.Color.fromBytes(r, g, b, alpha * 255);",
    "return Cesium.Color.fromBytes(r, g, b, Math.round(alpha * 255));"
)

with open("c:/KruthimaOps/production/app/static/app.js", "w", encoding="utf-8") as f:
    f.write(content)
print("app.js fixed")
