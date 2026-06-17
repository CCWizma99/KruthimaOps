import re

path = 'c:/KruthimaOps/production/app/static/app.js'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace in loadSubdivisions
content = content.replace(
    "ent.polygon.outlineWidth = 3;\n            ent.polygon.material = ent.polygon.material.color.getValue().withAlpha(0.2);",
    "ent.polygon.outlineWidth = 3;\n            ent.polygon.extrudedHeight = 500;\n            ent.polygon.material = ent.polygon.material.color.getValue().withAlpha(0.2);"
)

# Replace in restoreNationalView
content = content.replace(
    "ent.polygon.outline = false;\n      }\n      if (ent.label) ent.label.show = true;",
    "ent.polygon.outline = false;\n        ent.polygon.extrudedHeight = undefined;\n      }\n      if (ent.label) ent.label.show = true;"
)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Outline fix applied")
