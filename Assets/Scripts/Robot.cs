using System;
using UnityEngine;

public class Robot : MonoBehaviour
{
    #region Basic Stats
    public float moveSpeed = 4.0f;
    public float rotateSpeed = 200.0f;
    #endregion

    #region Dash Stats
    public float dashSpeed = 10.0f;
    public float dashDuration = 0.5f;       // Dash duration. 
    public float dashCooldown = 1.0f;       // Dash cooldown.
    private float lastDashTime = 0.0f;
    private bool isDashing = false;
    #endregion

    #region Physics Stats
    private Rigidbody2D rb;
    private Vector2 lastVelocity = Vector2.zero;
    public float stopDelay = 0.5f;           // Time before robot stops.
    public float slowDownRate = 2.0f;        // Robot's slowdown rate (velocity and rotation decay). 
    private float lastActTime = 0.0f;        // Last input or collision time, excluding rotation. 
    #endregion

    void Start()
    {
        // Get RigidBody2D.
        rb = GetComponent<Rigidbody2D>();
        
        // Ensure physics are handled manually. 
        rb.gravityScale = 0;        // Disable gravity for top-down movement
        rb.linearDamping = 0;       // No drag, we handle it manually
        rb.angularDamping = 0;      // No automatic angular slow-down

        // Set last interaction time to current time.
        lastActTime = Time.time;
    }

    void Update()
    {
        HandleInput();
        UpdateDashState();
        HandleStopping();
    }

    void HandleInput()
    {
        float rotation = 0.0f;
        bool hasInput = false;

        // Handle Robot rotation 
        if (Input.GetKey(KeyCode.LeftArrow))
        {
            rotation = rotateSpeed * Time.deltaTime;
            //[Todo] should modify the forward speed using current rb direction. 
        }
        else if (Input.GetKey(KeyCode.RightArrow))
        {
            rotation = -rotateSpeed * Time.deltaTime;
            //[Todo] should modify the forward speed using current rb direction.
        }
        transform.Rotate(0, 0, rotation);
        //[Todo] This makes the bounce logic wrong. Check the time of last collision.
        if (rb.linearVelocity.magnitude >= moveSpeed) {
            rb.linearVelocity = rb.linearVelocity.magnitude * transform.up;
            Debug.Log(rb.linearVelocity.magnitude);
        }
        //
        

        // Handle Robot movement and dash
        if (Input.GetKey(KeyCode.UpArrow)) 
        {
            float speed = isDashing ? dashSpeed : moveSpeed; //[Todo] This could be redundant with the Dash(). 
            rb.linearVelocity = transform.up * speed;
            hasInput = true;
        }
        if (Input.GetKeyDown(KeyCode.Space) && Time.time >= lastDashTime + dashCooldown) 
        {
            Dash();
            hasInput = true;
        }

        // Reset last action time if there is input (move and dash), rotation excluded.
        if (hasInput)
        {
            lastActTime = Time.time;
        }
    }

    void Dash()
    { 
        isDashing = true;
        lastDashTime = Time.time;
        rb.linearVelocity = transform.up * dashSpeed;
    }

    //[Todo] Change to UpdateState() to be more general. 
    void UpdateDashState()
    {
        if (isDashing && Time.time >= lastDashTime + dashDuration)
        {
            isDashing = false;
        }
        lastVelocity = rb.linearVelocity;
        //[Debug]
        if (isDashing && Input.GetKey(KeyCode.LeftArrow)) {
            Debug.Log("IsDashing? " + isDashing);
        } 
    }

    private void OnCollisionEnter2D(Collision2D collision)
    {
        GameObject otherObject = collision.gameObject;

        if (collision.gameObject.CompareTag("Robot"))
        { 
            if (otherObject.TryGetComponent<Robot>(out Robot otherRobot))
            { 
                // Calculate bounce direction 
                Vector2 collisionNormal = collision.contacts[0].normal; //[Todo] GetContact()
                Vector2 bounceDirection = Vector2.Reflect(lastVelocity.normalized, collisionNormal);
                
                float impactForce = lastVelocity.magnitude;
                otherRobot.Bounce(bounceDirection, impactForce);
                lastActTime = Time.time;
            }
        }
    }

    public void Bounce(Vector2 direction, float force)
    { 
        rb.linearVelocity = direction * force;
        //[Todo] Need to handle after get hit from dashing enemy, uncontrollable for short time. 
    }

    void HandleStopping()
    {
        if (Time.time > lastActTime + stopDelay)
        {
            // Gradually decrease linear and angular velocities 
            rb.linearVelocity = Vector2.Lerp(rb.linearVelocity, Vector2.zero, slowDownRate * Time.deltaTime); //[Todo] Need to just stop after it close to zero.
            rb.angularVelocity = Mathf.Lerp(rb.angularVelocity, 0, slowDownRate * Time.deltaTime);
        }
    }
}
