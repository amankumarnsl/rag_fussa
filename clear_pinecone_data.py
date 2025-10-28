#!/usr/bin/env python3
"""
Temporary script to clear all embeddings from Pinecone database
This will delete ALL vectors and data from the Pinecone index
"""

import os
import asyncio
from pinecone import Pinecone

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Get Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "rag-single-namespace")

async def clear_all_pinecone_data():
    """Clear all data from Pinecone index"""
    try:
        print("🔧 Connecting to Pinecone...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX)
        
        print(f"📊 Getting current index stats...")
        stats_before = index.describe_index_stats()
        total_vectors_before = stats_before.total_vector_count
        print(f"   Total vectors before: {total_vectors_before}")
        
        if total_vectors_before == 0:
            print("✅ Index is already empty - nothing to delete")
            return
        
        print("⚠️  WARNING: This will delete ALL vectors from the index!")
        print(f"   Index: {PINECONE_INDEX}")
        print(f"   Vectors to delete: {total_vectors_before}")
        
        # Confirm deletion
        confirm = input("\n❓ Are you sure you want to delete ALL data? (type 'YES' to confirm): ")
        if confirm != "YES":
            print("❌ Operation cancelled")
            return
        
        print("🗑️  Deleting all vectors...")
        
        # Method 1: Try to delete all vectors by deleting everything
        try:
            # Delete all vectors (this should work for most cases)
            index.delete(delete_all=True)
            print("✅ All vectors deleted successfully")
        except Exception as e:
            print(f"⚠️  Method 1 failed: {e}")
            print("🔄 Trying alternative method...")
            
            # Method 2: Delete by namespace (if using namespaces)
            try:
                # Get all namespaces and delete each one
                stats = index.describe_index_stats()
                namespaces = stats.get('namespaces', {})
                
                if namespaces:
                    for namespace_name in namespaces.keys():
                        print(f"   Deleting namespace: {namespace_name}")
                        index.delete(delete_all=True, namespace=namespace_name)
                else:
                    # If no namespaces, try deleting without namespace
                    index.delete(delete_all=True)
                
                print("✅ All vectors deleted successfully (alternative method)")
            except Exception as e2:
                print(f"❌ Alternative method also failed: {e2}")
                print("💡 You may need to manually clear the index from Pinecone console")
                return
        
        # Verify deletion
        print("🔍 Verifying deletion...")
        stats_after = index.describe_index_stats()
        total_vectors_after = stats_after.total_vector_count
        
        print(f"📊 Final stats:")
        print(f"   Vectors before: {total_vectors_before}")
        print(f"   Vectors after: {total_vectors_after}")
        
        if total_vectors_after == 0:
            print("✅ SUCCESS: All data cleared from Pinecone!")
        else:
            print(f"⚠️  WARNING: {total_vectors_after} vectors still remain")
            print("   You may need to manually clear the index from Pinecone console")
        
    except Exception as e:
        print(f"❌ Error clearing Pinecone data: {e}")
        print("💡 Check your Pinecone API key and index name")

if __name__ == "__main__":
    print("🚀 Pinecone Data Cleanup Script")
    print("=" * 50)
    asyncio.run(clear_all_pinecone_data())
    print("=" * 50)
    print("🏁 Script completed")
