using System.Collections.Generic;
using System.Linq;
using CoreSumoRobot;
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

        public string action;


        public EA_MCTS_Node(EA_MCTS_Node parent, string action)
        {
            this.parent = parent;
            this.action = action;
        }

        public EA_MCTS_Node Select()
        {
            if (children.Count < AIBot_EA_MCTS.PossibleActions.Keys.Count)
            {
                // Expand new child
                var name = AIBot_EA_MCTS.PossibleActions.ToList()[children.Count].Key;
                EA_MCTS_Node child = new EA_MCTS_Node(this, name);
                child.name = name;
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

        public float Simulate(SumoRobotController enemy, SumoRobotController controller, float simulationTime)
        {
            Vector3 aiDirection = controller.transform.up;
            Vector3 aiPosition = controller.transform.position;

            // Simulate action effect (approximate)
            if (action != null)
            {
                if (action.Equals("TurnLeftAction"))
                    aiDirection = Quaternion.Euler(0, 0, 45f) * aiDirection;
                else if (action.Equals("TurnRightAction"))
                    aiDirection = Quaternion.Euler(0, 0, -45f) * aiDirection;
                else if (action.Equals("DashAction"))
                    aiPosition += aiDirection.normalized * controller.DashSpeed * simulationTime;
                else if (action.Equals("AccelerateAction"))
                    aiPosition += aiDirection.normalized * controller.MoveSpeed * simulationTime;
                else if (action.Equals("SkillAction"))
                {
                    if (controller.Skill.Type == ERobotSkillType.Boost)
                        aiPosition += aiDirection.normalized * controller.MoveSpeed * simulationTime;
                }
            }

            Vector3 toEnemy = enemy.transform.position - aiPosition;
            float distance = toEnemy.magnitude;
            float angle = Vector3.SignedAngle(aiDirection, toEnemy.normalized, Vector3.forward);

            float angleScore = Mathf.Cos(angle * Mathf.Deg2Rad);
            float distScore = 1f - Mathf.Clamp01(distance / 3f);

            return angleScore + distScore;
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