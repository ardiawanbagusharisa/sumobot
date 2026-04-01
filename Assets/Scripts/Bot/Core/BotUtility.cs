using System;
using System.Collections.Generic;
using System.Linq;
using SumoBot;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

public class BotUtility
{
    private static List<Type> cached;
    private static List<Bot> cachedBotInstances;

    public static List<Type> GetAllBotTypes()
    {
        cached ??= AppDomain.CurrentDomain.GetAssemblies()
                .SelectMany(a => a.GetTypes())
                .Where(t => typeof(Bot).IsAssignableFrom(t)
                            && !t.IsAbstract
                            && t.IsSubclassOf(typeof(ScriptableObject)))
                .OrderBy(t => t.Name)
                .ToList();
        return cached;
    }

    public static List<Bot> GetAllBotInstances()
    {
        if (cachedBotInstances != null && cachedBotInstances.Count > 0)
            return cachedBotInstances;

        cachedBotInstances = new List<Bot>();

        // In Build: Try loading from Resources folder
        cachedBotInstances = Resources.LoadAll<Bot>("").ToList();

        // Fallback: create from types if no resources found
        if (cachedBotInstances.Count == 0)
        {
            var botTypes = GetAllBotTypes();
            foreach (var type in botTypes)
            {
                Bot bot = ScriptableObject.CreateInstance(type) as Bot;
                if (bot != null)
                {
                    cachedBotInstances.Add(bot);
                }
            }
        }

        // Sort by ID
        cachedBotInstances = cachedBotInstances.OrderBy(bot => bot.ID).ToList();

        return cachedBotInstances;
    }
}