using System;
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

            var bounceImpactResistA = robotB.GetComponent<SumoRobot>().BounceResistance;
            var bounceImpactResistB = robotA.GetComponent<SumoRobot>().BounceResistance;

            if (vA == 0 && robotB.LastRobotSkillType == ERobotSkillType.Stone)
            {
                bounceB = vB / total;
            }

            if (vB == 0 && robotA.LastRobotSkillType == ERobotSkillType.Stone)
            {
                bounceA = vA / total;
            }

            bounceA *= bounceImpactResistA;
            bounceB *= bounceImpactResistB;

            robotA.Bounce(collisionNormal, bounceA);       // away from B
            robotB.Bounce(-collisionNormal, bounceB);      // away from A

            Debug.Log($"[BOUNCE] {robotA.GetComponent<SumoRobot>().IdInt}(v={vA}) -> {bounceA}, {robotB.GetComponent<SumoRobot>().IdInt}(v={vB}) -> {bounceB}");
        }
    }
}