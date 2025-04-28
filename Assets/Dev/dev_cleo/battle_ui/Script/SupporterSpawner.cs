using System.Collections.Generic;
using UnityEngine;

public class SupporterSpawner : MonoBehaviour
{
    [Header("Setup Area")]
    public Transform spawnAreaObject;           // Objek kotak biasa (bisa punya Collider atau ukurannya manual)
    public GameObject[] supporterPrefabs;       // Prefab supporter
    public int spawnCount = 10;                  // Berapa banyak supporter yang muncul

    [Header("Spacing")]
    public float minDistanceBetween = 1f;        // Jarak minimum antar supporter

    private List<GameObject> currentSupporters = new List<GameObject>();
    private List<Vector3> usedPositions = new List<Vector3>();
    private Bounds spawnBounds;

    void Start()
    {
        // Dapatkan bounds dari objek
        CalculateBounds();
        SpawnSupporters();
    }

    private void CalculateBounds()
    {
        Collider2D col = spawnAreaObject.GetComponent<Collider2D>();
        if (col != null)
        {
            spawnBounds = col.bounds;
        }
        else
        {
            // Kalau nggak ada collider, pakai ukuran manual
            spawnBounds = new Bounds(spawnAreaObject.position, new Vector3(5f, 5f, 0f)); // <-- EDIT ukuran default kalau mau
        }
    }

    public void SpawnSupporters()
    {
        ClearSupporters();
        usedPositions.Clear();

        for (int i = 0; i < spawnCount; i++)
        {
            Vector3 pos = GetRandomNonOverlappingPosition();
            GameObject prefab = supporterPrefabs[Random.Range(0, supporterPrefabs.Length)];

            GameObject supporter = Instantiate(prefab, pos, Quaternion.identity);
            supporter.transform.SetParent(spawnAreaObject, true); // true supaya mempertahankan local transform

            // Tidak mengubah localScale apapun lagi!
            // supporter.transform.localScale = supporterSize; --> ini DILANGKAHI

            FloatingSupporter mover = supporter.AddComponent<FloatingSupporter>();
            mover.Init(spawnAreaObject, spawnBounds.size, minDistanceBetween);


            currentSupporters.Add(supporter);
            usedPositions.Add(supporter.transform.position);
        }
    }


    private Vector3 GetRandomNonOverlappingPosition()
    {
        int maxAttempts = 100;
        for (int attempt = 0; attempt < maxAttempts; attempt++)
        {
            float x = Random.Range(spawnBounds.min.x, spawnBounds.max.x);
            float y = Random.Range(spawnBounds.min.y, spawnBounds.max.y);
            Vector3 candidate = new Vector3(x, y, spawnAreaObject.position.z);

            bool tooClose = false;
            foreach (Vector3 pos in usedPositions)
            {
                if (Vector3.Distance(candidate, pos) < minDistanceBetween)
                {
                    tooClose = true;
                    break;
                }
            }

            if (!tooClose)
                return candidate;
        }

        // Fallback
        return new Vector3(
            Random.Range(spawnBounds.min.x, spawnBounds.max.x),
            Random.Range(spawnBounds.min.y, spawnBounds.max.y),
            spawnAreaObject.position.z
        );
    }

    public void ClearSupporters()
    {
        foreach (GameObject supporter in currentSupporters)
        {
            if (supporter != null)
                Destroy(supporter);
        }
        currentSupporters.Clear();
    }
}
