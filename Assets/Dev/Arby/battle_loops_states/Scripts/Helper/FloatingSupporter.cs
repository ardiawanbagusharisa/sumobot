using UnityEngine;

public class FloatingSupporter : MonoBehaviour
{
    private Transform trans;
    private Transform areaTransform;
    private Vector3 areaSize;
    private Vector3 targetPos;
    private float spacing = 1f;

    public float speed = 0.5f;

    public void Init(Transform areaTrans, Vector3 areaWorldSize, float minSpacing)
    {
        trans = GetComponent<Transform>();
        areaTransform = areaTrans;
        areaSize = areaWorldSize;
        spacing = minSpacing;

        PickNewTarget();
    }

    void Update()
    {
        if (trans == null || areaTransform == null)
            return;

        trans.position = Vector3.MoveTowards(trans.position, targetPos, speed * Time.deltaTime);

        if (Vector3.Distance(trans.position, targetPos) < 0.1f)
        {
            PickNewTarget();
        }
    }

    private void PickNewTarget()
    {
        if (areaTransform == null) return;

        float halfX = areaSize.x / 2f;
        float halfY = areaSize.y / 2f;

        float x = Random.Range(areaTransform.position.x - halfX + spacing, areaTransform.position.x + halfX - spacing);
        float y = Random.Range(areaTransform.position.y - halfY + spacing, areaTransform.position.y + halfY - spacing);

        targetPos = new Vector3(x, y, trans.position.z);
    }
}