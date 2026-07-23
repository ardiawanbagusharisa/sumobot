using UnityEngine;

/// <summary>
/// Device-level game settings (audio + language), persisted with PlayerPrefs. Kept separate
/// from SumoServices.PlayerData (the account save) because these preferences belong to the
/// machine, not the account.
/// </summary>
public static class GameSettings
{
    private const string MusicVolumeKey = "sumo.settings.musicVolume";
    private const string SfxVolumeKey = "sumo.settings.sfxVolume";
    private const string MusicMutedKey = "sumo.settings.musicMuted";
    private const string SfxMutedKey = "sumo.settings.sfxMuted";
    private const string LanguageKey = "sumo.settings.language";

    public const string DefaultLanguage = "English";

    // Languages offered in the dropdown; the picker builds itself from this list. No real
    // localization is wired yet, so selecting one only stores the preference.
    public static readonly string[] SupportedLanguages = { "English", "Chinese(todo)" };

    /// <summary>Music bus volume, 0..1 linear. Ignores mute — see <see cref="EffectiveMusicVolume"/>.</summary>
    public static float MusicVolume
    {
        get => Mathf.Clamp01(PlayerPrefs.GetFloat(MusicVolumeKey, 1f));
        set { PlayerPrefs.SetFloat(MusicVolumeKey, Mathf.Clamp01(value)); PlayerPrefs.Save(); }
    }

    /// <summary>Sound-effects bus volume, 0..1 linear. Ignores mute — see <see cref="EffectiveSfxVolume"/>.</summary>
    public static float SfxVolume
    {
        get => Mathf.Clamp01(PlayerPrefs.GetFloat(SfxVolumeKey, 1f));
        set { PlayerPrefs.SetFloat(SfxVolumeKey, Mathf.Clamp01(value)); PlayerPrefs.Save(); }
    }

    public static bool MusicMuted
    {
        get => PlayerPrefs.GetInt(MusicMutedKey, 0) == 1;
        set { PlayerPrefs.SetInt(MusicMutedKey, value ? 1 : 0); PlayerPrefs.Save(); }
    }

    public static bool SfxMuted
    {
        get => PlayerPrefs.GetInt(SfxMutedKey, 0) == 1;
        set { PlayerPrefs.SetInt(SfxMutedKey, value ? 1 : 0); PlayerPrefs.Save(); }
    }

    /// <summary>Selected UI language (one of <see cref="SupportedLanguages"/>).</summary>
    public static string Language
    {
        get => PlayerPrefs.GetString(LanguageKey, DefaultLanguage);
        set { PlayerPrefs.SetString(LanguageKey, string.IsNullOrEmpty(value) ? DefaultLanguage : value); PlayerPrefs.Save(); }
    }

    /// <summary>Volume the Music bus should actually play at (0 when muted).</summary>
    public static float EffectiveMusicVolume => MusicMuted ? 0f : MusicVolume;

    /// <summary>Volume the SFX bus should actually play at (0 when muted).</summary>
    public static float EffectiveSfxVolume => SfxMuted ? 0f : SfxVolume;
}
