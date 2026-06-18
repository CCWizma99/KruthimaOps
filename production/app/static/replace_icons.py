import re

path = 'c:/KruthimaOps/production/app/static/index.html'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

replacements = [
    ('🌊', '<i data-lucide="waves" class="icon-lg"></i>'),
    ('🔄', '<i data-lucide="refresh-cw" class="icon-sm"></i>'),
    ('📍 District', '<i data-lucide="map-pin" class="icon-sm"></i> District'),
    ('🗺️ Map Layers', '<i data-lucide="layers" class="icon-sm"></i> Map Layers'),
    ('🗺️ Open Evacuation Module', '<i data-lucide="map" class="icon-sm" style="margin-right:8px"></i> Open Evacuation Module'),
    ('⚡ Simulate Scenario', '<i data-lucide="zap" class="icon-sm" style="margin-right:8px"></i> Simulate Scenario'),
    ('🕰️ Time Travel', '<i data-lucide="clock" class="icon-sm"></i> Time Travel'),
    ('⏪ Run Historical', '<i data-lucide="rewind" class="icon-sm" style="margin-right:6px"></i> Run Historical'),
    ('🕰️', '<i data-lucide="history" class="icon-lg"></i>'),
    ('↩ Return to Live', '<i data-lucide="corner-up-left" class="icon-sm" style="margin-right:4px"></i> Return to Live'),
    ('📊 Model Registry', '<i data-lucide="bar-chart-2" class="icon-sm"></i> Model Registry'),
    ('📤 Batch Upload', '<i data-lucide="upload-cloud" class="icon-sm"></i> Batch Upload'),
    ('📋 Activity Log', '<i data-lucide="list" class="icon-sm"></i> Activity Log'),
    ('📅 7-Day Risk Forecast', '<i data-lucide="calendar" class="icon-sm"></i> 7-Day Risk Forecast'),
    ('🤖 AI Safety Briefing', '<i data-lucide="cpu" class="icon-sm"></i> AI Safety Briefing'),
    ('🛡️ Nearest Safe Zone', '<i data-lucide="shield" class="icon-sm"></i> Nearest Safe Zone'),
    ('👍', '<i data-lucide="thumbs-up" class="icon-sm"></i>'),
    ('👎', '<i data-lucide="thumbs-down" class="icon-sm"></i>'),
    ('⚡ Scenario Simulator', '<i data-lucide="zap" class="icon-sm"></i> Scenario Simulator'),
    ('⚡ Simulate', '<i data-lucide="zap" class="icon-sm"></i> Simulate'),
    ('⚡ Analyse Risk', '<i data-lucide="activity" class="icon-sm"></i> Analyse Risk'),
    ('📄 Report', '<i data-lucide="file-text" class=\"icon-sm\"></i> Report'),
    ('📄 Download PDF Report', '<i data-lucide="file-text" class="icon-sm"></i> Download PDF Report')
]

for old, new in replacements:
    content = content.replace(old, new)

if '<script src="https://unpkg.com/lucide@latest"></script>' not in content:
    content = content.replace('</head>', '  <script src="https://unpkg.com/lucide@latest"></script>\n</head>')

if 'lucide.createIcons()' not in content:
    content = content.replace('</body>', '  <script>lucide.createIcons();</script>\n</body>')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print("Replaced index.html")
