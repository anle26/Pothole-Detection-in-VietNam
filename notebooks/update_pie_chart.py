"""Update the pie chart cells in the notebook with new grouping and explanation."""
import json
import os

notebook_path = os.path.join(
    os.path.dirname(__file__),
    "03_explore_vietnam_dataset.ipynb"
)

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Remove the previously added pie chart cells if they exist
cells = nb.get("cells", [])
if len(cells) >= 2 and cells[-2].get("id") == "pie_chart_header" and cells[-1].get("id") == "pie_chart_code":
    nb["cells"] = cells[:-2]

# --- Updated Markdown cell ---
md_cell = {
    "cell_type": "markdown",
    "id": "pie_chart_header",
    "metadata": {},
    "source": [
        "### 3. Phân bố ảnh theo Đường (Street) và theo Split (Train/Valid/Test)\n",
        "Biểu đồ tròn thể hiện trong 500 tấm ảnh thuộc tập `VietNamDataset`, mỗi ảnh thuộc đường nào và nằm trong folder đầu vào nào.\n",
        "\n",
        "**Ghi chú quan trọng về phân bố dữ liệu:** Dữ liệu được lựa chọn tập trung chủ yếu vào các tuyến đường chính và gom các đường nhỏ khác thành nhóm **\"Đường hỗn hợp\"**. Đây là tập dữ liệu mang tính chủ đích, thể hiện những đoạn đường đang thi công hoặc xuống cấp, có nhiều ổ gà mang tính **đặc trưng cho điều kiện giao thông và thị trường Việt Nam**, chứ không tập trung vào đường đất đá nông thôn."
    ]
}

# --- Updated Code cell: Pie chart ---
code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "pie_chart_code",
    "metadata": {},
    "outputs": [],
    "source": [
        "# ========================================================\n",
        "# Biểu đồ tròn: Phân bố 500 ảnh theo Nhóm Đường và theo Split\n",
        "# ========================================================\n",
        "import os\n",
        "import glob\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "\n",
        "DATASET_PATH = \"../data/VietNamDataset\"\n",
        "\n",
        "def extract_base_id(filename):\n",
        "    \"\"\"Trích xuất ID gốc (bỏ hậu tố Roboflow)\"\"\"\n",
        "    basename = os.path.splitext(os.path.basename(filename))[0]\n",
        "    if \"_jpg\" in basename:\n",
        "        return basename.split(\"_jpg\")[0]\n",
        "    return basename\n",
        "\n",
        "# 1. Thu thập tên ảnh từ các folder gốc (đường/street)\n",
        "EXCLUDE = {'train', 'valid', 'test', 'full_dataset'}\n",
        "source_map = {}  # base_id -> street_folder_name\n",
        "\n",
        "for item in sorted(os.listdir(DATASET_PATH)):\n",
        "    full = os.path.join(DATASET_PATH, item)\n",
        "    if os.path.isdir(full) and item not in EXCLUDE:\n",
        "        for f in os.listdir(full):\n",
        "            if f.lower().endswith(('.jpg', '.png', '.jpeg')):\n",
        "                base_id = os.path.splitext(f)[0]\n",
        "                source_map[base_id] = item\n",
        "\n",
        "# 2. Mapping từng ảnh trong train/valid/test -> Nhóm đường gốc\n",
        "per_split = {}  # split -> {group: count}\n",
        "overall = {}    # group -> total_count\n",
        "\n",
        "MAIN_STREETS = ['DinhCongTrangStreet', 'HoangHuuNamStreet', 'National_Highway_1']\n",
        "\n",
        "for split in ['train', 'valid', 'test']:\n",
        "    img_path = os.path.join(DATASET_PATH, split, 'images')\n",
        "    if not os.path.exists(img_path):\n",
        "        continue\n",
        "    per_split[split] = {}\n",
        "    for f in os.listdir(img_path):\n",
        "        if not f.lower().endswith(('.jpg', '.png', '.jpeg')):\n",
        "            continue\n",
        "        base_id = extract_base_id(f)\n",
        "        folder = source_map.get(base_id, 'Không xác định')\n",
        "        \n",
        "        # Phân loại vào nhóm chính hoặc \"Đường hỗn hợp\"\n",
        "        if folder in MAIN_STREETS:\n",
        "            group = folder\n",
        "        else:\n",
        "            group = 'Đường hỗn hợp'\n",
        "            \n",
        "        per_split[split][group] = per_split[split].get(group, 0) + 1\n",
        "        overall[group] = overall.get(group, 0) + 1\n",
        "\n",
        "total = sum(overall.values())\n",
        "print(f\"Tổng số ảnh trong train/valid/test: {total}\\n\")\n",
        "\n",
        "# ===========================\n",
        "# BIỂU ĐỒ TRÒN TỔNG THỂ\n",
        "# ===========================\n",
        "\n",
        "# Đổi tên hiển thị cho đẹp\n",
        "DISPLAY_NAMES = {\n",
        "    'DinhCongTrangStreet': 'Đinh Công Tráng',\n",
        "    'HoangHuuNamStreet': 'Hoàng Hữu Nam',\n",
        "    'National_Highway_1': 'Quốc lộ 1',\n",
        "    'Đường hỗn hợp': 'Đường hỗn hợp'\n",
        "}\n",
        "\n",
        "labels_overall = []\n",
        "sizes_overall = []\n",
        "for group in sorted(overall.keys(), key=lambda x: overall[x], reverse=True):\n",
        "    disp_name = DISPLAY_NAMES.get(group, group)\n",
        "    labels_overall.append(f\"{disp_name}\\n({overall[group]})\")\n",
        "    sizes_overall.append(overall[group])\n",
        "\n",
        "colors = plt.cm.Pastel1(np.linspace(0, 1, len(labels_overall)))\n",
        "\n",
        "fig, axes = plt.subplots(1, 2, figsize=(18, 7))\n",
        "\n",
        "# -- Pie chart tổng thể --\n",
        "wedges, texts, autotexts = axes[0].pie(\n",
        "    sizes_overall, labels=labels_overall, autopct='%1.1f%%',\n",
        "    colors=colors, startangle=140, pctdistance=0.8,\n",
        "    wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'},\n",
        "    textprops={'fontsize': 11, 'fontweight': 'bold'}\n",
        ")\n",
        "axes[0].set_title(\n",
        "    f'Phân bố {total} ảnh theo Nhóm Đường (tổng thể)',\n",
        "    fontsize=14, fontweight='bold', pad=20\n",
        ")\n",
        "\n",
        "# -- Stacked bar chart theo split --\n",
        "split_names = ['train', 'valid', 'test']\n",
        "unique_groups = sorted(overall.keys(), key=lambda x: overall[x], reverse=True)\n",
        "x = np.arange(len(split_names))\n",
        "bottom = np.zeros(len(split_names))\n",
        "\n",
        "for idx, group in enumerate(unique_groups):\n",
        "    vals = [per_split.get(s, {}).get(group, 0) for s in split_names]\n",
        "    disp_name = DISPLAY_NAMES.get(group, group)\n",
        "    axes[1].bar(x, vals, bottom=bottom, label=disp_name,\n",
        "                color=colors[idx], edgecolor='white', linewidth=1)\n",
        "    bottom += vals\n",
        "\n",
        "axes[1].set_xticks(x)\n",
        "axes[1].set_xticklabels([f'{s.upper()}\\n({sum(per_split.get(s, {}).values())})' for s in split_names],\n",
        "                         fontsize=12, fontweight='bold')\n",
        "axes[1].set_ylabel('Số lượng ảnh', fontsize=12)\n",
        "axes[1].set_title('Phân bố theo Split và Nhóm Đường', fontsize=14, fontweight='bold')\n",
        "axes[1].legend(loc='upper right', fontsize=10, title='Nhóm Đường', title_fontsize=11)\n",
        "\n",
        "# Thêm số liệu lên bar\n",
        "for i, s in enumerate(split_names):\n",
        "    total_split = sum(per_split.get(s, {}).values())\n",
        "    axes[1].text(i, total_split + 5, str(total_split),\n",
        "                ha='center', fontweight='bold', fontsize=11)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.savefig('eda_outputs/pie_chart_street_distribution.png', dpi=150, bbox_inches='tight')\n",
        "plt.show()\n"
    ]
}

# Append updated cells
nb["cells"].append(md_cell)
nb["cells"].append(code_cell)

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Updated notebook successfully!")
