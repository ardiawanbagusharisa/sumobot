using System.Collections.Generic;
using UnityEngine;

namespace RobotCoreAction
{
    public class RobotStats : MonoBehaviour
    {


        #region Basic Stats
        public float MoveSpeed = 4.0f;
        public float RotateSpeed = 200.0f;
        #endregion

        #region Dash Stats
        public float DashSpeed = 10.0f;
        public float DashDuration = 0.5f;       // Dash duration. 
        public float DashCooldown = 1.0f;       // Dash cooldown.
        #endregion

        #region Physics Stats
        public float StopDelay = 0.5f;           // Time before robot stops.
        public float SlowDownRate = 2.0f;        // Robot's slowdown rate (velocity and rotation decay). 
        #endregion

        // this is historical
        public Dictionary<ERobotActionType, float> ActionsTime = new Dictionary<ERobotActionType, float>();


    }
}