using System;
using System.Collections.Generic;
using System.Linq;
using SumoBot;
using UnityEngine;

public class BotUtility
{
    private static List<Type> cached;
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
}