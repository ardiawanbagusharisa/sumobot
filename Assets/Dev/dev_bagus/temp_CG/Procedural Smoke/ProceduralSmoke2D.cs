using UnityEngine;
using System.Collections.Generic;

public class ProceduralSmoke2D : MonoBehaviour
{
    [Header("Smoke Settings")]
    public GameObject smokePrefab;
    public int poolSize = 50;

    [Header("Emission Settings")]
    public float emissionRate = 10f;
    public float smokeLifetime = 2f;
    public Vector2 initialSize = new Vector2(0.2f, 0.2f);
    public Vector2 finalSize = new Vector2(1f, 1f);

    [Header("Randomness Settings")]
    public float spawnRadius = 0.2f;
    public float randomDrift = 0.5f;
    public float upwardSpeed = 0.5f;

    [Header("Rotation Settings")]
    public float minRotationSpeed = -90f;  // Degrees per second
    public float maxRotationSpeed = 90f;

    [Header("Wind Settings")]
    public Vector2 windDirection = new Vector2(0.2f, 0f);

    [Header("Damage State Colors")]
    public Color normalSmokeStart = new Color(0.5f, 0.5f, 0.5f, 0.8f);
    public Color normalSmokeEnd = new Color(0.5f, 0.5f, 0.5f, 0f);
    public Color heavySmokeStart = new Color(0.2f, 0.2f, 0.2f, 0.9f);
    public Color heavySmokeEnd = new Color(0.1f, 0.1f, 0.1f, 0f);
    public Color fireSmokeStart = new Color(1f, 0.5f, 0f, 0.9f);
    public Color fireSmokeEnd = new Color(1f, 0.2f, 0f, 0f);

    private List<SmokeParticle> pool = new();
    private float emissionTimer = 0f;
    private System.Func<Color> startColorFunc;
    private System.Func<Color> endColorFunc;

    public enum DamageState
    {
        Normal,
        HeavySmoke,
        Fire
    }
    public DamageState currentDamageState = DamageState.Normal;

    private void Start()
    {
        for (int i = 0; i < poolSize; i++)
        {
            GameObject obj = Instantiate(smokePrefab, transform);
            obj.SetActive(false);
            pool.Add(new SmokeParticle { obj = obj, spriteRenderer = obj.GetComponent<SpriteRenderer>() });
        }
    }

    private void Update()
    {
        EmitSmoke();
        UpdateSmokeParticles();
    }

    void EmitSmoke()
    {
        emissionTimer += Time.deltaTime;
        float timePerParticle = 1f / emissionRate;

        while (emissionTimer > timePerParticle)
        {
            emissionTimer -= timePerParticle;
            SpawnSmokeParticle();
        }
    }

    void SpawnSmokeParticle()
    {
        SmokeParticle p = GetAvailableParticle();
        if (p == null) return; // Pool exhausted

        Vector2 spawnOffset = Random.insideUnitCircle * spawnRadius;
        p.obj.transform.position = transform.position + (Vector3)spawnOffset;
        p.obj.transform.localScale = initialSize;
        p.lifetime = 0f;

        // Choose color based on damage state
        switch (currentDamageState)
        {
            case DamageState.Normal:
                p.startColor = normalSmokeStart;
                p.endColor = normalSmokeEnd;
                break;
            case DamageState.HeavySmoke:
                p.startColor = heavySmokeStart;
                p.endColor = heavySmokeEnd;
                break;
            case DamageState.Fire:
                p.startColor = fireSmokeStart;
                p.endColor = fireSmokeEnd;
                break;
        }

        p.spriteRenderer.color = p.startColor;
        p.velocity = new Vector2(Random.Range(-randomDrift, randomDrift), upwardSpeed + Random.Range(0f, randomDrift));
        p.rotationSpeed = Random.Range(minRotationSpeed, maxRotationSpeed);
        p.obj.transform.rotation = Quaternion.Euler(0, 0, Random.Range(0f, 360f));

        p.obj.SetActive(true);
    }

    void UpdateSmokeParticles()
    {
        foreach (var p in pool)
        {
            if (!p.obj.activeSelf) continue;

            p.lifetime += Time.deltaTime;
            if (p.lifetime > smokeLifetime)
            {
                p.obj.SetActive(false);
                continue;
            }

            // Move with wind and velocity
            Vector2 totalVelocity = p.velocity + windDirection;
            p.obj.transform.position += (Vector3)(totalVelocity * Time.deltaTime);
            p.obj.transform.Rotate(0f, 0f, p.rotationSpeed * Time.deltaTime);
            float t = p.lifetime / smokeLifetime;
            if (p.spriteRenderer != null)
            {
                p.spriteRenderer.color = Color.Lerp(p.startColor, p.endColor, t);
                p.obj.transform.localScale = Vector2.Lerp(initialSize, finalSize, t);
            }
        }
    }

    SmokeParticle GetAvailableParticle()
    {
        foreach (var p in pool)
        {
            if (!p.obj.activeSelf)
                return p;
        }
        return null;
    }

    private class SmokeParticle
    {
        public GameObject obj;
        public Vector2 velocity;
        public float lifetime;
        public Color startColor;
        public Color endColor;
        public float rotationSpeed;

        public SpriteRenderer spriteRenderer;
    }

    // Optional: API to change damage state at runtime
    public void SetDamageState(DamageState newState)
    {
        currentDamageState = newState;
    }
}
