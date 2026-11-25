using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace SumoCampaign
{
    public static class CampaignCatalog
    {
        private static IReadOnlyList<CampaignLevelDefinition> levels;
        public static IReadOnlyList<CampaignLevelDefinition> Levels => levels ??=
            Resources.LoadAll<CampaignLevelDefinition>("Campaign")
                .OrderBy(level => level.Id, StringComparer.Ordinal).ToArray();

        public static CampaignLevelDefinition Selected { get; private set; }

        public static void Launch(CampaignLevelDefinition level)
        {
            if (level == null) throw new ArgumentNullException(nameof(level));
            Selected = level;
            SceneManager.LoadScene(level.Scene);
        }

        public static void ClearSelection() => Selected = null;
    }
}
