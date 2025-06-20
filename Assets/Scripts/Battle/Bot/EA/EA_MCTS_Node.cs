using System;
using System.Collections.Generic;
using System.Linq;
using CoreSumo;
using UnityEngine;

namespace BotAI
{
    public class EA_MCTS_Node
    {
        public string name;
        public EA_MCTS_Node parent;
        public List<EA_MCTS_Node> children = new List<EA_MCTS_Node>();
        public int visits = 0;
        public float totalReward = 0f;

        public ISumoAction action;
        public string actionString;
        public ISumoAction badAction;
        public ISumoAction goodAction;
        public bool badActionAlreadyUsed = false;
        public bool goodActionAlreadyUsed = false;


        public EA_MCTS_Node(EA_MCTS_Node parent, ISumoAction action, ISumoAction goodAction = null, ISumoAction badAction = null)
        {
            this.parent = parent;
            this.action = action;
            this.badAction = badAction;
            this.goodAction = goodAction;
            totalReward = 0f;
            visits = 0;
            children.Clear();
        }

        public EA_MCTS_Node Select()
        {
            if (children.Count < AIBot_EA_MCTS.PossibleActions.Count)
            {
                // Expand new child
                var possibleAct = AIBot_EA_MCTS.PossibleActions.ToList()[children.Count];

                EA_MCTS_Node child = new EA_MCTS_Node(this, possibleAct);

                if (!badActionAlreadyUsed && badAction != null && possibleAct == badAction)
                {
                    child.action = badAction;
                    child.totalReward = -1f;
                    badActionAlreadyUsed = true;
                }

                if (!goodActionAlreadyUsed && goodAction != null && possibleAct == goodAction)
                {
                    child.action = goodAction;
                    child.totalReward = 1f;
                    badActionAlreadyUsed = true;
                }

                child.name = possibleAct.GetType().Name;
                children.Add(child);
                return child;
            }

            // UCB1 selection
            float logParentVisits = Mathf.Log(visits + 1);
            EA_MCTS_Node best = null;
            float bestValue = float.MinValue;
            foreach (var child in children)
            {
                float ucb1 = (child.totalReward / (child.visits + 1e-4f)) +
                             1.41f * Mathf.Sqrt(logParentVisits / (child.visits + 1e-4f));
                if (ucb1 > bestValue)
                {
                    bestValue = ucb1;
                    best = child;
                }
            }
            return best.Select();
        }

        public float Simulate(SumoController enemy, SumoController controller, float simulationTime)
        {
            Vector3 aiDirection = controller.transform.up;
            Vector3 aiPosition = controller.transform.position;

            // Simulate action effect (approximate)
            if (action is TurnLeftAngleAction)
            {
                aiDirection = Quaternion.Euler(0, 0, (float)action.Param) * aiDirection;
            }
            else if (action is TurnRightAngleAction rV)
            {
                aiDirection = Quaternion.Euler(0, 0, (float)action.Param) * aiDirection;
            }
            else if (action is AccelerateAction)
            {
                aiPosition += aiDirection.normalized * controller.MoveSpeed * simulationTime;
            }
            else if (action is DashAction && !controller.IsDashOnCooldown)
            {
                aiPosition += aiDirection.normalized * controller.DashSpeed * controller.DashDuration;
            }
            else if (action is SkillAction && !controller.Skill.IsSkillCooldown)
            {
                if (controller.Skill.Type == ERobotSkillType.Boost)
                    aiPosition += aiDirection.normalized * (controller.MoveSpeed * controller.Skill.BoostMultiplier) * simulationTime;
            }

            GameObject arena = BattleManager.Instance.Arena;
            float arenaRadius = arena.GetComponent<CircleCollider2D>().radius * arena.transform.lossyScale.x;
            Vector3 arenaCenter = arena.transform.position;

            Vector3 toEnemy = enemy.transform.position - aiPosition;
            float distance = toEnemy.magnitude;
            float angle = Vector3.SignedAngle(aiDirection, toEnemy.normalized, Vector3.forward);

            float angleScore = Mathf.Cos(angle * Mathf.Deg2Rad);
            float distScore = 1f - Mathf.Clamp01(distance / arenaRadius);

            float bonusParam = 0;
            bool IsPossibleOutFromArena = Vector3.Distance(aiPosition, arenaCenter) > arenaRadius;
            if (IsPossibleOutFromArena && (action is AccelerateAction || action is DashAction || action is SkillAction))
            {
                // Penalize heavily if sumo will exits the ring, or any action that makes the Sumo move away from exits, reward instead.
                bonusParam = (distScore - 0.7f) * 30f;

                Debug.Log($"[Simulate][IsPossibleOutFromArena] {action.GetType().Name}, param: {action} can cause go outside of arena\n Detail:Vector3.Distance({aiPosition}, {arenaCenter}) > {arenaRadius}, resulting: {bonusParam}, distScore: {distScore}");
            }

            return angleScore + distScore + bonusParam;
        }

        public void Backpropagate(float reward)
        {
            visits++;
            totalReward += reward;
            parent?.Backpropagate(reward);
        }

        public EA_MCTS_Node GetBestChild()
        {
            EA_MCTS_Node best = null;
            float bestScore = float.MinValue;
            foreach (var child in children)
            {
                float avg = child.totalReward / (child.visits + 1e-4f);
                if (avg > bestScore)
                {
                    bestScore = avg;
                    best = child;
                }
            }
            return best;
        }
    }
}