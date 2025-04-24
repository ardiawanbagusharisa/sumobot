using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class SupporterSpawner : MonoBehaviour
{
    [Header("Setup Area")]
    public RectTransform spawnArea;             // Panel UI tempat gambar akan muncul
    public GameObject[] supporterPrefabs;       // Prefab gambar (misal 60x60 UI Image)
    public int spawnCount = 10;                 // Jumlah gambar yang ingin di-spawn

    [Header("Spacing")]
    public float minDistanceBetween = 70f;      // Jarak minimum antar gambar (lebih besar dari 60)

    private List<GameObject> currentSupporters = new List<GameObject>();
    private List<Vector2> usedPositions = new List<Vector2>();

    void Start()
    {
        SpawnSupporters(); // Spawn otomatis saat Start
    }

    public void SpawnSupporters()
    {
        ClearSupporters();
        usedPositions.Clear();

        for (int i = 0; i < spawnCount; i++)
        {
            Vector2 pos = GetRandomNonOverlappingPosition();
            GameObject prefab = supporterPrefabs[Random.Range(0, supporterPrefabs.Length)];
            GameObject supporter = Instantiate(prefab, spawnArea);

            RectTransform rt = supporter.GetComponent<RectTransform>();
            if (rt != null)
            {
                rt.sizeDelta = new Vector2(60f, 60f);
                rt.anchoredPosition = pos;
                rt.localScale = Vector3.one;
            }

            // Tambahkan skrip gerakan
            FloatingSupporter mover = supporter.AddComponent<FloatingSupporter>();
            mover.Init(spawnArea, minDistanceBetween);

            currentSupporters.Add(supporter);
            usedPositions.Add(pos);
        }
    }

    private Vector2 GetRandomNonOverlappingPosition()
    {
        Vector2 size = spawnArea.rect.size;
        Vector2 candidate;
        int maxAttempts = 100;

        for (int attempt = 0; attempt < maxAttempts; attempt++)
        {
            float x = Random.Range(-size.x / 2f + 30f, size.x / 2f - 30f);
            float y = Random.Range(-size.y / 2f + 30f, size.y / 2f - 30f);
            candidate = new Vector2(x, y);

            bool tooClose = false;
            foreach (Vector2 pos in usedPositions)
            {
                if (Vector2.Distance(candidate, pos) < minDistanceBetween)
                {
                    tooClose = true;
                    break;
                }
            }

            if (!tooClose)
                return candidate;
        }

        // Fallback
        return new Vector2(
            Random.Range(-size.x / 2f + 30f, size.x / 2f - 30f),
            Random.Range(-size.y / 2f + 30f, size.y / 2f - 30f)
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
