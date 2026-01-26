// Assets/Editor/IgnoreMilvusVolumes.cs
using UnityEditor;
using System.IO;
using UnityEngine;

[InitializeOnLoad]
public class UnityMetaFilter
{
    static UnityMetaFilter()
    {
        // Set labels on the volumes folder to be ignored
        string[] volumeFolders = new string[]
        {
            "Assets/Scripts/Bot/Example/ML/LLM/volumes/etcd",
            "Assets/Scripts/Bot/Example/ML/LLM/volumes/milvus",
            "Assets/Scripts/Bot/Example/ML/LLM/volumes/minio"
        };

        foreach (string folder in volumeFolders)
        {
            if (Directory.Exists(folder))
            {
                // Get all files recursively
                string[] files = Directory.GetFiles(folder, "*.*", SearchOption.AllDirectories);
                foreach (string file in files)
                {
                    // Set each file to not be included in build
                    AssetDatabase.SetLabels(AssetDatabase.LoadAssetAtPath<Object>(file), new string[] { "ExcludeFromBuild" });
                }
            }
        }
    }
}