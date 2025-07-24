from comproscanner import ComProScanner

main_property_keyword = "piezoelectric"
property_keywords = {"exact_keywords": ["d33"], "substring_keywords": [" d 33 "]}


if __name__ == "__main__":
    comproscanner = ComProScanner(main_property_keyword=main_property_keyword)
    # comproscanner.collect_metadata()

    # comproscanner.process_articles(property_keywords=property_keywords)

    comproscanner.extract_composition_property_data(
        main_extraction_keyword="d33",
        json_results_file="deepseek/deepseek-v3-piezo-ceramic-test-results.json",
    )
