import os
import weaviate
from weaviate.classes.init import Auth

weaviate_client = weaviate.connect_to_local(
    auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY"))
)

print(f"Weaviate ready: {weaviate_client.is_ready()}")

if not weaviate_client.collections.exists("Documents"):
    weaviate_client.collections.create(
        "Documents",
        vector_config=[
            weaviate.classes.config.Configure.Vectors.text2vec_transformers(
                name="semantics_vector",
                source_properties=["semantics"]
            )
        ]
    )
    print("Created collection 'Documents'")
else:
    print("Collection already exists")

def db_get_file_nuggets(file_path: str):
    documents_collection = weaviate_client.collections.use("Documents")
    return documents_collection.query.fetch_objects(filters=weaviate.classes.query.Filter.by_property("file_path").equal(file_path))

def db_clear_file_nuggets(file_path: str):
    documents_collection = weaviate_client.collections.use("Documents")
    documents_collection.data.delete_many(where=weaviate.classes.query.Filter.by_property("file_path").like(file_path))

def db_insert_nuggets_if_not_exist(file_path: str, nuggets: list[str]) -> int:
    existing_nuggets = db_get_file_nuggets(file_path)
    if (len(existing_nuggets.objects) > 0):
        return 0
    
    documents_collection = weaviate_client.collections.use("Documents")

    with documents_collection.batch.fixed_size(batch_size=200) as batch:
        for nugget in nuggets:
            batch.add_object(
                properties={
                    "file_path": file_path,
                    "semantics": nugget,
                },
            )
            if batch.number_errors > 10:
                print("Batch import stopped due to excessive errors")
                documents_collection.data.delete_many(where=weaviate.classes.query.Filter.by_property("file_path").like(file_path))
                break
        
        batch.flush()

    return len(nuggets) - len(documents_collection.batch.failed_objects)

def db_query(query: str):
    documents_collection = weaviate_client.collections.use("Documents")
    response = documents_collection.query.near_text(query=query, return_metadata=weaviate.classes.query.MetadataQuery(distance=True, certainty=True), limit=5)
    return [item.properties for item in response.objects if item.metadata.distance < 0.75]