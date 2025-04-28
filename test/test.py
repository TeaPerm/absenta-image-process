import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from table_detector import detect_table_and_cells

test_cases = [
    # 1.
    {
        "image_path": "test/example-2.jpeg",
        "names_list": [
            "Ablakos Zsiga",
            "Bajusz Rudi",
            "Csutak Töhötöm",
            "Döcögő Bertók",
            "Esernyős Vidor",
            "Fityisz Ödön",
            "Gubanc Rezső",
            "Hablaty Zénó",
            "Iciri Pacal",
            "Jampi Lóránt",
            "Kacat Menyhért",
            "Lityi-Lötyi Béla",
            "Móka Mirkó",
            "Nyammogó Sári",
            "Ordítós Géza",
            "Pöttyös Dömötör",
            "Quargli Tivadar",
            "Röfögő Zoltán",
            "Szöszke Domokos",
            "Tökfilkó Ernő"
        ]
    },
    # 2.
    {
        "image_path": "test/example-3.jpeg",
        "names_list": [
            "Ablakos Zsiga",
            "Bajusz Rudi",
            "Csutak Töhötöm",
            "Döcögő Bertók",
            "Esernyős Vidor",
            "Fityisz Ödön",
            "Gubanc Rezső",
            "Hablaty Zénó",
            "Iciri Pacal",
            "Jampi Lóránt",
            "Kacat Menyhért",
            "Lityi-Lötyi Béla",
            "Móka Mirkó",
            "Nyammogó Sári",
            "Ordítós Géza",
            "Pöttyös Dömötör",
            "Quargli Tivadar",
            "Röfögő Zoltán",
            "Szöszke Domokos",
            "Tökfilkó Ernő"
        ]
    },
    # 3.
    {
        "image_path": "test/example8.jpg",
     "names_list": [
        "Abrisin Alen",
        "Boros Botond",
        "Consuegra-Sotolongo Gábor Luis",
        "Csontos Dávid Ferenc",
        "Dancs Kornél",
        "Ferenczy Kata",
        "Gombos Vidor Márton",
        "Hanyecz Rebeka",
        "Kelemen Kevin Tamás",
        "Le Thien Nam",
        "Mágó Szabolcs",
        "Németh Dávid",
        "Péter Dávid",
        "Rakonczai Soma",
        "Simon Raffael",
        "Takács Máté",
        "Tóth Izabella",
        "Tóth Levente",
        "Török Bálint Bence",
        "Valtai Domonkos"
    ]
    },
    # 4.
    {
        "image_path": "test/example7.jpg",
     "names_list": [
        "Abrisin Alen",
        "Boros Botond",
        "Consuegra-Sotolongo Gábor Luis",
        "Csontos Dávid Ferenc",
        "Dancs Kornél",
        "Ferenczy Kata",
        "Gombos Vidor Márton",
        "Hanyecz Rebeka",
        "Kelemen Kevin Tamás",
        "Le Thien Nam",
        "Mágó Szabolcs",
        "Németh Dávid",
        "Péter Dávid",
        "Rakonczai Soma",
        "Simon Raffael",
        "Takács Máté",
        "Tóth Izabella",
        "Tóth Levente",
        "Török Bálint Bence",
        "Valtai Domonkos"
    ]
    },
    # 5.
    {
        "image_path": "test/sablon_example_5.jpg",
     "names_list": [
        "Abrisin Alen",
        "Boros Botond",
        "Consuegra-Sotolongo Gábor Luis",
        "Csontos Dávid Ferenc",
        "Dancs Kornél",
        "Ferenczy Kata",
        "Gombos Vidor Márton",
        "Hanyecz Rebeka",
        "Kelemen Kevin Tamás",
        "Le Thien Nam",
        "Mágó Szabolcs",
        "Németh Dávid",
        "Péter Dávid",
        "Rakonczai Soma",
        "Simon Raffael",
        "Takács Máté",
        "Tóth Izabella",
        "Tóth Levente",
        "Török Bálint Bence",
        "Valtai Domonkos"
    ]
    },
]

for idx, case in enumerate(test_cases, 1):
    print(f"\n--- Running test case {idx}: {case['image_path']} ---")
    output_image, results = detect_table_and_cells(case["image_path"], case["names_list"])
    test_dir = f"test_output/test_{idx}"
    os.makedirs(test_dir, exist_ok=True)

    # Save JSON results
    json_path = os.path.join(test_dir, "result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {json_path}")

    # Save output image
    if output_image is not None:
        out_path = os.path.join(test_dir, "output_table.jpg")
        import cv2
        cv2.imwrite(out_path, output_image)
        print(f"Output image saved to {out_path}")

    # Print error if present
    if results and "message" in results:
        print(f"Error: {results['message']}")
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))