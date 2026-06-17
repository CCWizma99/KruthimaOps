import re

with open("c:/KruthimaOps/production/app/static/app.js", "r", encoding="utf-8") as f:
    content = f.read()

# I will find the function and replace it fully
new_func = """function getInterpolatedColor(score, alpha) {
  let r = 0, g = 0, b = 0;
  if (score < 0.25) {
    const t = score / 0.25;
    r = Math.round(34 + t * (234 - 34));
    g = Math.round(197 + t * (179 - 197));
    b = Math.round(94 + t * (8 - 94));
  } else if (score < 0.50) {
    const t = (score - 0.25) / 0.25;
    r = Math.round(234 + t * (249 - 234));
    g = Math.round(179 + t * (115 - 179));
    b = Math.round(8 + t * (22 - 8));
  } else if (score < 0.75) {
    const t = (score - 0.50) / 0.25;
    r = Math.round(249 + t * (239 - 249));
    g = Math.round(115 + t * (68 - 115));
    b = Math.round(22 + t * (68 - 22));
  } else {
    r = 239;
    g = 68;
    b = 68;
  }
  return Cesium.Color.fromBytes(r, g, b, Math.round(alpha * 255));
}"""

# regex to replace the function
content = re.sub(r'function getInterpolatedColor\(score, alpha\)\s*\{.*?\n\}', new_func, content, flags=re.DOTALL)

with open("c:/KruthimaOps/production/app/static/app.js", "w", encoding="utf-8") as f:
    f.write(content)
print("app.js fixed")
