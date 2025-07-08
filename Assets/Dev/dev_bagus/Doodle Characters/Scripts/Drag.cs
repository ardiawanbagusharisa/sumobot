using UnityEngine;

public class Drag : MonoBehaviour
{
    public enum PlayerSide { 
        Left, 
        Right 
    }
    public PlayerSide side = PlayerSide.Left;

    public float yLagAmount = 1f;
    public float ySmoothTime = 0.5f;
    public float rotationLagAmount = 30f;
    public float rotationSmoothTime = 0.2f;

    private float yVelocity;
    private float rotationVelocity;
    private float targetRotation;

    void Update()
    {
        float moveX = 0f;
        float moveY = 0f;

        // --- Input based on player side ---
        if (side == PlayerSide.Left)
        {
            moveX = Input.GetKey(KeyCode.D) ? 1f : Input.GetKey(KeyCode.A) ? -1f : 0f;
            moveY = Input.GetKey(KeyCode.W) ? 1f : 0f;
        }
        else if (side == PlayerSide.Right)
        {
            moveX = Input.GetKey(KeyCode.K) ? 1f : Input.GetKey(KeyCode.Semicolon) ? -1f : 0f;
            moveY = Input.GetKey(KeyCode.O) ? 1f : 0f;
        }

        // --- Flip logic enabled here ---
        //float newLag = yLagAmount * (side == PlayerSide.Left ? 1f : -1f);
        float targetLocalY = moveY > 0 ? yLagAmount : 0f;
        Vector3 localPos = transform.localPosition;
        localPos.y = Mathf.SmoothDamp(localPos.y, targetLocalY, ref yVelocity, ySmoothTime);
        transform.localPosition = localPos;

        if (Mathf.Abs(moveX) > 0.01f)
        {
            targetRotation = (side == PlayerSide.Left ? 1f : -1f) * Mathf.Sign(moveX) * rotationLagAmount;
        }
        else
        {
            targetRotation = 0f;
        }

        float currentZ = transform.localEulerAngles.z;
        if (currentZ > 180f) currentZ -= 360f;
        float newZ = Mathf.SmoothDamp(currentZ, targetRotation, ref rotationVelocity, rotationSmoothTime);

        Vector3 newEuler = transform.localEulerAngles;
        newEuler.z = newZ;
        transform.localEulerAngles = newEuler;
    }
}
