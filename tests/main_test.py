from comproscanner.metadata_extractor.fetch_metadata import FetchMetadata
from comproscanner.metadata_extractor.filter_metadata import FilterMetadata
from comproscanner.article_processors.elsevier_processor import ElsevierArticleProcessor

main_property_keyword = "piezoelectric"
base_queries = ["piezoelectric"]
extra_queries = ["advancements"]
property_keywords = {
    "exact_keywords": ["d33"],
    "substring_keywords": [" d 33 "],
}

if __name__ == "__main__":
    # fetch_metadata = FetchMetadata(
    #     main_property_keyword=main_property_keyword,
    #     base_queries=base_queries,
    #     extra_queries=extra_queries,
    #     end_year=2024,
    # )
    # fetch_metadata.main_fetch()

    # filter_metadata = FilterMetadata(main_property_keyword=main_property_keyword)
    # filter_metadata.filter_metadata()

    elsevier_processor = ElsevierArticleProcessor(
        main_property_keyword=main_property_keyword,
        property_keywords=property_keywords,
        start_row=1,
        end_row=100,
    )
    elsevier_processor.process_elsevier_articles()
