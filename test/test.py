import sys
import os
import json
import argparse
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
        "image_path": "test/test-5.jpeg",
        "names_list": [
     "Almási Dóra",
    "Bíró Gellért",
    "Csernai László",
    "Dancsó Mira",
    "Erdősi Balázs",
    "Farkas Zsófi",
    "Gál Benedek",
    "Hegyi Petra",
    "Iványi Máté",
    "Juhász Noémi",
    "Kovács Richárd",
    "Lukács Enikő",
    "Molnár Zoltán",
    "Nagy Emese",
    "Oláh Csaba",
    "Papp Klaudia",
    "Quintusz Bálint",
    "Rácz Lili",
    "Szabó Viktor",
    "Tóth Fruzsina",
    "Urbán Levente",
    "Varga Patrícia",
    "Wágner Attila",
    "Zentai Nikolett"
]

    },
    # 3.
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
    # 4.
    {
        "image_path": "test/test-5-2.jpeg",
        "names_list": [
     "Almási Dóra",
    "Bíró Gellért",
    "Csernai László",
    "Dancsó Mira",
    "Erdősi Balázs",
    "Farkas Zsófi",
    "Gál Benedek",
    "Hegyi Petra",
    "Iványi Máté",
    "Juhász Noémi",
    "Kovács Richárd",
    "Lukács Enikő",
    "Molnár Zoltán",
    "Nagy Emese",
    "Oláh Csaba",
    "Papp Klaudia",
    "Quintusz Bálint",
    "Rácz Lili",
    "Szabó Viktor",
    "Tóth Fruzsina",
    "Urbán Levente",
    "Varga Patrícia",
    "Wágner Attila",
    "Zentai Nikolett"
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
    # 6.
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

def run_test(test_idx):
    """Run a specific test case by index (1-based)"""
    if test_idx < 1 or test_idx > len(test_cases):
        print(f"Error: Test index must be between 1 and {len(test_cases)}")
        return
        
    case = test_cases[test_idx - 1]
    print(f"\n--- Running test case {test_idx}: {case['image_path']} ---")
    output_image, results = detect_table_and_cells(case["image_path"], case["names_list"])
    test_dir = f"test_output/test_{test_idx}"
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

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run table detection tests")
    parser.add_argument(
        "test_idx", 
        type=int, 
        nargs="?", 
        default=0,
        help="Index of the test to run (1-6). If not provided, runs all tests."
    )
    
    args = parser.parse_args()
    
    if args.test_idx == 0:
        # Run all tests
        for idx in range(1, len(test_cases) + 1):
            run_test(idx)
    else:
        # Run a specific test
        run_test(args.test_idx)