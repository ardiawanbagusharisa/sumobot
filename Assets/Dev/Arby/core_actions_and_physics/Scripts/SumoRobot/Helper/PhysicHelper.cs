using System;
using BattleLoop;
using UnityEngine;

namespace CoreSumoRobot
{
    public class PhysicHelper
    {
        public static void HandleBounce(SumoRobotController robotA, SumoRobotController robotB, Vector2 collisionNormal, float baseForce = 4f)
        {
            SumoRobot sumoA = robotA.GetComponent<SumoRobot>();
            SumoRobot sumoB = robotB.GetComponent<SumoRobot>();

            float vA = robotA.LastVelocity.magnitude;
            float vB = robotB.LastVelocity.magnitude;

            float total = vA + vB + 0.01f;

            float bounceA = baseForce * (vB / total);  // robotA gets more bounce if B has more speed
            float bounceB = baseForce * (vA / total);  // robotB gets more bounce if A has more speed

            float bounceImpactResistA = sumoB.BounceResistance;
            float bounceImpactResistB = sumoA.BounceResistance;

            if (vA == 0 && robotB.sumoSkill.CurrentSkillType == ERobotSkillType.Stone)
            {
                bounceB = vB / total;
            }

            if (vB == 0 && robotA.sumoSkill.CurrentSkillType == ERobotSkillType.Stone)
            {
                bounceA = vA / total;
            }

            bounceA *= bounceImpactResistA;
            bounceB *= bounceImpactResistB;

            robotA.Bounce(collisionNormal, bounceA);       // away from B
            robotB.Bounce(-collisionNormal, bounceB);      // away from A

            BattleManager.Instance.CurrentRound.SetEventLog($"type=bounce;bouncerId={sumoA.IdInt};receiverId={sumoB.IdInt};bouncerImpact={bounceA};receiverImpact={bounceB}");
        }
    }
}