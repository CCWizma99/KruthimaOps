import re

path = 'c:/KruthimaOps/production/evacuation/evacuation_presentation.html'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

emojis = set(re.findall(r'[^\x00-\x7F]+', content))

replacements = [
    ('🗺️', '<i data-lucide="map" class="icon-sm"></i>'),
    ('🛡️', '<i data-lucide="shield" class="icon-sm"></i>'),
    ('📍', '<i data-lucide="map-pin" class="icon-sm"></i>'),
    ('🟢', '<i data-lucide="wifi" class="icon-sm"></i>'),
    ('🔴', '<i data-lucide="wifi-off" class="icon-sm"></i>'),
    ('⬇', '<i data-lucide="download" class="icon-sm"></i>'),
    ('📦', '<i data-lucide="package" class="icon-sm"></i>'),
    ('📌', '<i data-lucide="map-pin" class="icon-sm"></i>'),
    ('🛰️', '<i data-lucide="satellite" class="icon-sm"></i>'),
    ('➕', '<i data-lucide="plus" class="icon-sm"></i>'),
    ('✓', '<i data-lucide="check" class="icon-sm"></i>'),
    ('❌', '<i data-lucide="x" class="icon-sm"></i>'),
    ('🚨', '<i data-lucide="alert-triangle" class="icon-sm"></i>'),
    ('🔍', '<i data-lucide="search" class="icon-sm"></i>'),
    ('⚙', '<i data-lucide="settings" class="icon-sm"></i>'),
    ('—', '&mdash;'),
    ('🌊', '<i data-lucide="waves" class="icon-sm"></i>')
]

for old, new in replacements:
    content = content.replace(old, new)

if '<script src="https://unpkg.com/lucide@latest"></script>' not in content:
    content = content.replace('</head>', '  <script src="https://unpkg.com/lucide@latest"></script>\n</head>')

# We must render icons dynamically since Leaflet or other JS might add them later
# Just putting lucide.createIcons() at the bottom
if 'lucide.createIcons()' not in content:
    content = content.replace('</body>', '  <script>lucide.createIcons();</script>\n</body>')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Done")
