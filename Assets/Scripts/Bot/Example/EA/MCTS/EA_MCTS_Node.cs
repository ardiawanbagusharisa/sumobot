using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using SumoCore;
using UnityEngine;

namespace SumoBot
{
    public class EA_MCTS_Node
    {
        public string ID;
        public EA_MCTS_Node head;
        public EA_MCTS_Node parent;
        public List<EA_MCTS_Node> children = new();
        public int visits = 0;
        public float totalReward = 0f;
        public float angleScore = 0f;
        public float distScore = 0f;
        public float bonusOrPenalty = 0f;
        public List<ISumoAction> actions = new();
        public string actionString;
        public List<ISumoAction> badAction;
        public List<ISumoAction> goodAction;
        public bool badActionAlreadyUsed = false;
        public bool goodActionAlreadyUsed = false;
        public PlayerSide Side;

        public HighestScoreType GetHighestScoreType()
        {
            try
            {
                Dictionary<float, HighestScoreType> highestScore = new()
                {
                    {angleScore,HighestScoreType.Angle},
                    {distScore,HighestScoreType.Distance},
                    {bonusOrPenalty,HighestScoreType.BonusOrPenalty}
                };
                float result = highestScore.Max((i) => i.Key);
                return highestScore[result];
            }
            catch (Exception)
            {
                return HighestScoreType.Random;
            }
        }

        public EA_MCTS_Node(EA_MCTS_Node parent, List<ISumoAction> actions, List<ISumoAction> goodAction = null, List<ISumoAction> badAction = null)
        {
            this.parent = parent;
            this.actions = actions;
            this.badAction = badAction;
            this.goodAction = goodAction;
        }

        public void Init(Dictionary<string, EA_MCTS_Node> AllNodes)
        {
            children.Clear();
            totalReward = 0f;
            visits = 0;
            foreach (var action in actions)
            {
                EA_MCTS_Node newNode = new(this, new List<ISumoAction>() { action });
                newNode.ID = action.FullName;
                if (goodAction != null && goodAction.Count > 0)
                {
                    newNode.totalReward = 1;
                    newNode.visits = 1;
                }
                if (badAction != null && badAction.Count > 0)
                {
                    newNode.totalReward = -1;
                    newNode.visits = 1;
                }
                children.Add(newNode);
                AllNodes.Add(action.FullName, newNode);
            }
        }

        public EA_MCTS_Node Expand(Dictionary<string, EA_MCTS_Node> AllNodes)
        {
            var unexploredActs = AIBot_EA_MCTS.PossibleActions.Where(x =>
            {
                var newActNames = $"{ID}:{x.FullName}";
                if (!AllNodes.ContainsKey(newActNames))
                    return true;
                return false;
            }).ToList();

            if (unexploredActs.Count() == 0)
                return null;

            System.Random random = new();
            var randomAction = unexploredActs[random.Next(unexploredActs.Count())];
            string newActNames;
            newActNames = $"{ID}:{randomAction.FullName}";

            var newActs = new List<ISumoAction>(actions) { randomAction };

            EA_MCTS_Node newNode = new(this, newActs)
            {
                ID = newActNames
            };
            children.Add(newNode);
            AllNodes.Add(newActNames, newNode);
            return newNode;
        }

        public void SortAction()
        {
            actions.Sort((x, y) =>
            {
                if (x.FullName.ToLower().Contains("turn"))
                {
                    return -100;
                }
                return x.FullName.GetHashCode().CompareTo(y.FullName.GetHashCode());
            });
        }

        public EA_MCTS_Node Select()
        {
            if (children.Count == 0) return this;
            double C = 1.41;
            return children.OrderByDescending(child =>
            {
                if (child.visits == 0) return double.MaxValue;
                double exploitation = child.totalReward / child.visits;
                double exploration = C * Math.Sqrt(Math.Log(visits + 1) / child.visits);
                return exploitation + exploration;
            }).FirstOrDefault();
        }

        public Tuple<float, float, float> Simulate(SumoAPI api, AI_MCTS_Config config)
        {

            Vector3 aiRot = Vector3.zero;
            Vector3 aiPos = Vector3.zero;

            float bonusOrPenalty = 0;
            float angleScore = 0;
            float distScore = 0;

            foreach (var action in actions)
            {
                if (action is SkillAction)
                {
                    action.Type = config.DefaultSkillType == SkillType.Stone ? ActionType.SkillStone : ActionType.SkillBoost;
                }

                SimulateResultAPI result = api.Simulate(action);
                aiPos += result.Position;
                aiRot += result.Rotation;

                bool approachWithPosition = action is AccelerateAction || action is DashAction;
                float preAngleScore = api.Angle(oriPos: aiPos, oriRot: aiRot, normalized: true);
                float preDistScore = api.DistanceNormalized(oriPos: aiPos);

                angleScore += preAngleScore;
                distScore += preDistScore;

                if (action.Type == ActionType.SkillStone)
                {
                    if (api.CanExecute(action))
                    {
                        bonusOrPenalty += 2f;

                        if (api.DistanceNormalized(oriPos: aiPos) >= (api.BattleInfo.ArenaRadius * 0.75))
                        {
                            bonusOrPenalty += 2f;

                            if (api.EnemyRobot.Skill.Type == SkillType.Boost && api.EnemyRobot.Skill.IsActive)
                                bonusOrPenalty += 2f;

                        }
                        continue;
                    }
                    else
                    {
                        bonusOrPenalty -= 0.1f;
                    }
                }
                else if (action.Type == ActionType.SkillBoost)
                {
                    if (api.CanExecute(action))
                    {
                        bonusOrPenalty += 2f;
                    }
                    else
                    {
                        bonusOrPenalty -= 0.1f;
                    }
                }

                // Define reward for action type that is approaching enemy
                if (approachWithPosition)
                {
                    float avg = (preAngleScore + preAngleScore) / 2;
                    float distFromArena = api.DistanceFromArena();
                    float nearArenaRadius = api.BattleInfo.ArenaRadius * 0.8f;

                    if (distFromArena > nearArenaRadius)
                    {
                        float angleToArena = api.Angle(targetPos: api.BattleInfo.ArenaPosition, normalized: true);
                        Debug.Log($"robot near arena {distFromArena} > {nearArenaRadius}: angle {angleToArena}");
                        // If robot near arena radius by 70%, reward approachWithPosition action to make closer to the center of arena
                        if (angleToArena >= 0.7f)
                        {
                            bonusOrPenalty += 2f;
                        }
                        else
                        {
                            bonusOrPenalty -= 1f;
                        }
                    }

                    if (avg > 0.90f)
                    {
                        // Reward robot if Enemy is in front of face (by rotation and by distance)
                        bonusOrPenalty += 3f;

                        if (action.Type == ActionType.Dash)
                            bonusOrPenalty += 0.5f;
                        continue;
                    }
                    else if (preAngleScore > 0.90f && api.MyRobot.Skill.Type == SkillType.Boost && api.MyRobot.Skill.IsActive)
                    {
                        // Reward robot if Enemy is in front of face (by rotation and by boost-skill)
                        bonusOrPenalty += 2f;

                        if (action.Type == ActionType.Dash)
                            bonusOrPenalty += 0.5f;
                        continue;
                    }
                }
                else
                {
                    // Penalize robot from trying to rotate farther than before
                    float originalAngle = api.Angle(normalized: true);
                    if (originalAngle > preAngleScore)
                    {
                        bonusOrPenalty -= 1f;
                        continue;
                    }
                }
            }

            float normAngleScore = angleScore / actions.Count();
            float normDistScore = distScore / actions.Count();
            float normBonusOrPenalty = bonusOrPenalty / actions.Count();

            this.angleScore += normAngleScore;
            this.distScore += normBonusOrPenalty;
            this.bonusOrPenalty += normBonusOrPenalty;
            return Tuple.Create(normAngleScore, normDistScore, normBonusOrPenalty);
        }

        public void Backpropagate(Tuple<float, float, float> reward)
        {
            visits++;
            totalReward += reward.Item1 + reward.Item2 + reward.Item3;
            angleScore += reward.Item1;
            distScore += reward.Item2;
            bonusOrPenalty += reward.Item3;
            parent?.Backpropagate(reward);
        }

        public EA_MCTS_Node GetBestChild(List<EA_MCTS_Node> nodes = null)
        {
            if (children.Count == 0) return null;

            var highest = children.OrderByDescending(child =>
            {
                double exploitation = child.totalReward / (child.visits + double.Epsilon);
                return exploitation;
            }).FirstOrDefault();

            if (nodes == null)
            {
                nodes = new() { highest };
            }
            else
            {
                nodes.Add(highest);
            }

            if (highest.children.Count == 0)
            {
                if (nodes == null)
                    return highest;
                else
                {
                    return nodes.OrderByDescending(child =>
                    {
                        double exploitation = child.totalReward / (child.visits + double.Epsilon);
                        return exploitation;
                    }).FirstOrDefault();
                }
            }
            return highest.GetBestChild(nodes);
        }
    }

    public enum HighestScoreType
    {
        Angle,
        Distance,
        BonusOrPenalty,
        Random,
    }
}