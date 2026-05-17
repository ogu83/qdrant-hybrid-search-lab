"""
Generate sample_products.json — 500 consumer electronics product records.
Run once: python generate_products.py
Produces realistic product names, SKUs, descriptions, categories, prices.
"""
import json
import random
from pathlib import Path

random.seed(42)

SMARTPHONES = [
    ("iPhone 15 Pro Max", ["128GB", "256GB", "512GB", "1TB"], ["Black Titanium", "White Titanium", "Natural Titanium", "Blue Titanium"], 1199),
    ("iPhone 15 Pro", ["128GB", "256GB", "512GB"], ["Black Titanium", "White Titanium", "Natural Titanium", "Blue Titanium"], 999),
    ("iPhone 15", ["128GB", "256GB", "512GB"], ["Black", "Blue", "Green", "Yellow", "Pink"], 799),
    ("Samsung Galaxy S24 Ultra", ["256GB", "512GB", "1TB"], ["Titanium Black", "Titanium Gray", "Titanium Violet", "Titanium Yellow"], 1299),
    ("Samsung Galaxy S24+", ["256GB", "512GB"], ["Onyx Black", "Marble Gray", "Cobalt Violet", "Amber Yellow"], 999),
    ("Samsung Galaxy S24", ["128GB", "256GB"], ["Onyx Black", "Marble Gray", "Cobalt Violet"], 799),
    ("Google Pixel 9 Pro XL", ["128GB", "256GB", "512GB"], ["Obsidian", "Porcelain", "Hazel", "Rose Quartz"], 1099),
    ("Google Pixel 9 Pro", ["128GB", "256GB"], ["Obsidian", "Porcelain", "Hazel"], 999),
    ("Google Pixel 9", ["128GB", "256GB"], ["Obsidian", "Porcelain", "Wintergreen"], 799),
    ("OnePlus 12", ["256GB", "512GB"], ["Silky Black", "Flowy Emerald"], 799),
]

LAPTOPS = [
    ("MacBook Pro 14-inch M3 Pro", ["18GB RAM 512GB", "18GB RAM 1TB", "36GB RAM 1TB"], ["Space Black", "Silver"], 1999),
    ("MacBook Pro 14-inch M3", ["8GB RAM 512GB", "8GB RAM 1TB"], ["Space Gray", "Silver"], 1599),
    ("MacBook Air 15-inch M3", ["8GB RAM 256GB", "8GB RAM 512GB", "16GB RAM 512GB"], ["Midnight", "Starlight", "Space Gray", "Silver"], 1299),
    ("MacBook Air 13-inch M3", ["8GB RAM 256GB", "8GB RAM 512GB", "16GB RAM 512GB"], ["Midnight", "Starlight", "Space Gray", "Silver"], 1099),
    ("Dell XPS 15 9530", ["16GB RAM 512GB RTX 4060", "32GB RAM 1TB RTX 4070"], ["Platinum Silver", "Graphite"], 1899),
    ("Dell XPS 13 Plus 9320", ["16GB RAM 512GB", "32GB RAM 1TB"], ["Platinum Silver", "Graphite"], 1299),
    ("Lenovo ThinkPad X1 Carbon Gen 12", ["16GB RAM 512GB", "32GB RAM 1TB"], ["Black"], 1599),
    ("ASUS ROG Zephyrus G14", ["16GB RAM 512GB RTX 4060", "32GB RAM 1TB RTX 4070"], ["Eclipse Gray", "Platinum White"], 1499),
    ("HP Spectre x360 14", ["16GB RAM 512GB", "32GB RAM 1TB"], ["Nightfall Black", "Natural Silver"], 1399),
    ("Microsoft Surface Laptop 6", ["16GB RAM 512GB", "32GB RAM 1TB"], ["Platinum", "Black", "Sapphire"], 1299),
]

TABLETS = [
    ("iPad Air 11-inch M2", ["128GB Wi-Fi", "256GB Wi-Fi", "128GB Wi-Fi + Cellular", "256GB Wi-Fi + Cellular"], ["Blue", "Purple", "Starlight", "Space Gray"], 599),
    ("iPad Air 13-inch M2", ["128GB Wi-Fi", "256GB Wi-Fi", "512GB Wi-Fi"], ["Blue", "Purple", "Starlight", "Space Gray"], 799),
    ("iPad Pro 11-inch M4", ["256GB Wi-Fi", "512GB Wi-Fi", "1TB Wi-Fi"], ["Space Black", "Silver"], 999),
    ("iPad Pro 13-inch M4", ["256GB Wi-Fi", "512GB Wi-Fi", "1TB Wi-Fi", "2TB Wi-Fi"], ["Space Black", "Silver"], 1299),
    ("Samsung Galaxy Tab S9 Ultra", ["256GB Wi-Fi", "512GB Wi-Fi"], ["Graphite", "Beige"], 1099),
    ("Samsung Galaxy Tab S9+", ["256GB Wi-Fi", "512GB Wi-Fi"], ["Graphite", "Beige"], 899),
    ("Samsung Galaxy Tab S9 FE", ["128GB Wi-Fi", "256GB Wi-Fi"], ["Gray", "Mint", "Lavender", "Silver"], 449),
    ("Google Pixel Tablet", ["128GB", "256GB"], ["Hazel", "Porcelain", "Rose"], 499),
]

HEADPHONES = [
    ("Sony WH-1000XM5", ["Single"], ["Black", "Silver", "Midnight Blue"], 349),
    ("Sony WH-1000XM4", ["Single"], ["Black", "Silver", "Midnight Blue"], 249),
    ("Bose QuietComfort 45", ["Single"], ["Black", "White Smoke", "Cypress Green"], 279),
    ("Bose QuietComfort Ultra", ["Single"], ["Black", "White Smoke", "Sandstone"], 429),
    ("Apple AirPods Max", ["Single"], ["Space Gray", "Silver", "Sky Blue", "Pink", "Starlight"], 549),
    ("AirPods Pro (2nd Gen) USB-C", ["Single"], ["White"], 249),
    ("AirPods Pro (2nd Gen) Lightning", ["Single"], ["White"], 229),
    ("AirPods (3rd Gen)", ["Single"], ["White"], 169),
    ("Samsung Galaxy Buds3 Pro", ["Single"], ["White", "Black"], 249),
    ("Jabra Evolve2 85", ["Single"], ["Black", "Beige"], 449),
]

STORAGE = [
    ("Samsung 990 Pro 2TB NVMe SSD", ["Single"], ["Black"], 199),
    ("Samsung 990 Pro 1TB NVMe SSD", ["Single"], ["Black"], 119),
    ("WD Black SN850X 2TB NVMe SSD", ["Single"], ["Black"], 179),
    ("WD Black SN850X 1TB NVMe SSD", ["Single"], ["Black"], 99),
    ("Samsung T7 Shield 2TB Portable SSD", ["Single"], ["Beige", "Black", "Blue"], 149),
    ("Samsung T7 Shield 1TB Portable SSD", ["Single"], ["Beige", "Black", "Blue"], 89),
    ("Seagate One Touch 5TB Portable HDD", ["Single"], ["Black", "Silver", "Red", "Blue"], 99),
    ("WD My Passport 4TB Portable HDD", ["Single"], ["Black", "Red", "Blue", "White"], 79),
]

TELEVISIONS = [
    ("LG C4 65-inch OLED 4K TV", ["Single"], ["Black"], 1699),
    ("LG C4 55-inch OLED 4K TV", ["Single"], ["Black"], 1199),
    ("Samsung QN90D 65-inch Neo QLED 4K TV", ["Single"], ["Titan Black"], 1799),
    ("Samsung QN90D 55-inch Neo QLED 4K TV", ["Single"], ["Titan Black"], 1299),
    ("Sony X90L 65-inch 4K TV", ["Single"], ["Black"], 1299),
    ("TCL QM8 65-inch QLED 4K TV", ["Single"], ["Black"], 999),
]

SMARTWATCHES = [
    ("Apple Watch Series 10 45mm GPS", ["Single"], ["Jet Black", "Rose Gold", "Silver", "Gold"], 429),
    ("Apple Watch Series 10 42mm GPS", ["Single"], ["Jet Black", "Rose Gold", "Silver", "Gold"], 399),
    ("Apple Watch Ultra 2", ["Single"], ["Black", "Natural Titanium", "White"], 799),
    ("Samsung Galaxy Watch 7 44mm", ["Single"], ["Green", "Cream", "Silver"], 299),
    ("Samsung Galaxy Watch 7 40mm", ["Single"], ["Green", "Cream", "Silver"], 249),
    ("Google Pixel Watch 3 45mm", ["Single"], ["Matte Black", "Porcelain", "Hazel"], 349),
]

PERIPHERALS = [
    ("Logitech MX Master 3S", ["Single"], ["Graphite", "Pale Gray", "Performance Red"], 99),
    ("Logitech MX Keys S", ["Single"], ["Graphite", "Pale Gray", "Rose"], 109),
    ("Apple Magic Mouse", ["Single"], ["Silver", "Space Gray", "Black"], 79),
    ("Apple Magic Keyboard with Touch ID", ["Single"], ["Silver", "Space Gray", "Black"], 99),
    ("Razer DeathAdder V3 Pro", ["Single"], ["Black", "White"], 149),
    ("Keychron Q1 Pro", ["Single"], ["Carbon Black", "Shell White"], 199),
]

CATEGORY_MAP = {
    "smartphone": SMARTPHONES,
    "laptop": LAPTOPS,
    "tablet": TABLETS,
    "headphones": HEADPHONES,
    "storage": STORAGE,
    "tv": TELEVISIONS,
    "smartwatch": SMARTWATCHES,
    "peripheral": PERIPHERALS,
}

DESCRIPTIONS = {
    "smartphone": "A flagship {name} featuring a powerful processor, advanced camera system, and premium build quality. Available in {storage} storage configuration in {color}.",
    "laptop": "The {name} delivers exceptional performance for professionals and creators. Configured with {storage} for demanding workloads. Available in {color}.",
    "tablet": "The {name} combines portability with powerful performance. Features a stunning Liquid Retina display and comes in {storage} configuration. Color: {color}.",
    "headphones": "Premium {name} with industry-leading noise cancellation technology, comfortable over-ear design, and exceptional audio quality. Available in {color}.",
    "storage": "High-performance {name} for fast data transfer and reliable storage. Ideal for professionals and power users who need speed and capacity.",
    "tv": "The {name} delivers stunning picture quality with advanced display technology, HDR support, and smart TV features for an immersive viewing experience.",
    "smartwatch": "The {name} tracks health metrics, delivers notifications, and keeps you connected throughout the day. Available in {color}.",
    "peripheral": "The {name} provides precision control and ergonomic comfort for productivity and creative work. Available in {color}.",
}


def generate_products() -> list[dict]:
    products = []
    pid = 0
    target = 500

    category_items = list(CATEGORY_MAP.items())

    while len(products) < target:
        cat_name, items = random.choice(category_items)
        base_name, storage_options, colors, base_price = random.choice(items)
        storage = random.choice(storage_options)
        color = random.choice(colors)

        if storage != "Single":
            full_name = f"{base_name} {storage}"
        else:
            full_name = base_name

        price = base_price + random.randint(-50, 50)
        description = DESCRIPTIONS[cat_name].format(
            name=base_name, storage=storage, color=color
        )

        sku = f"{cat_name[:3].upper()}-{pid:04d}"

        products.append({
            "id": pid,
            "name": full_name,
            "description": description,
            "category": cat_name,
            "color": color,
            "storage": storage,
            "price": price,
            "sku": sku,
            "text": f"{full_name} {description}",
        })
        pid += 1

    return products[:target]


if __name__ == "__main__":
    products = generate_products()
    out_path = Path(__file__).parent / "sample_products.json"
    with open(out_path, "w") as f:
        json.dump(products, f, indent=2)
    print(f"Generated {len(products)} products -> {out_path}")
