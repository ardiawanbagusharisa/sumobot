using System.Collections.Generic;
using UnityEngine;

namespace CoreSumoRobot
{
    public class SumoRobot : MonoBehaviour
    {
        #region Basic Stats
        public string Id => System.Guid.NewGuid().ToString();
        public int IdInt;
        public float MoveSpeed = 4.0f;
        public float RotateSpeed = 200.0f;
        #endregion

        #region Dash Stats
        public float DashSpeed = 5.0f;
        public float DashDuration = 0.5f;       // Dash duration. 
        public float DashCooldown = 1.0f;       // Dash cooldown.
        #endregion

        #region Physics Stats
        public float StopDelay = 0.5f;           // Time before robot stops.
        public float SlowDownRate = 2.0f;        // Robot's slowdown rate (velocity and rotation decay). 
        public float BounceResistance = 1f;
        #endregion
    }
}
