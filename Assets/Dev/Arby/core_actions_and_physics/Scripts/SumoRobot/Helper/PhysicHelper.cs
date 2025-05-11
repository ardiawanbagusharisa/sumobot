using System;
using BattleLoop;
using UnityEngine;

namespace CoreSumoRobot
{
    public class PhysicHelper
    {
        public static void HandleBounce(SumoRobotController robotA, SumoRobotController robotB, Vector2 collisionNormal, float baseForce = 5f)
        {

            float vA = robotA.LastVelocity.magnitude;
            float vB = robotB.LastVelocity.magnitude;

            float total = vA + vB + 0.01f;

            float bounceA = baseForce * (vB / total);  // robotA gets more bounce if B has more speed
            float bounceB = baseForce * (vA / total);  // robotB gets more bounce if A has more speed

            if (robotA.Skill.Type == ERobotSkillType.Stone && robotA.Skill.IsActive)
            {
                bounceB = vB / total;
            }

            if (robotB.Skill.Type == ERobotSkillType.Stone && robotA.Skill.IsActive)
            {
                bounceA = vA / total;
            }

            bounceA *= robotB.BounceResistance;
            bounceB *= robotA.BounceResistance;

            robotA.Bounce(collisionNormal, bounceA);       // away from B
            robotB.Bounce(-collisionNormal, bounceB);      // away from A

            Debug.Log($"[PhysicHelper] vA=>{vA} vB=>{vB} skillA={robotA.Skill.Type} skillB={robotB.Skill.Type} impactA=>{bounceA} impactB=>{bounceB}");
        }
    }
}