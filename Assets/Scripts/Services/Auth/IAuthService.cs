using System.Threading.Tasks;

namespace SumoServices
{
    /// <summary>
    /// Authentication boundary. The game talks to this interface only; the concrete
    /// implementation (UGS for the game, local only for legacy migration) is chosen
    /// inside GameServices. All methods are async because real backends do network I/O.
    /// </summary>
    public interface IAuthService
    {
        /// <summary>The currently signed-in account, or null when signed out.</summary>
        PlayerAccount Current { get; }

        bool IsSignedIn { get; }

        /// <summary>
        /// Sign in without credentials (device-bound / anonymous). This is the
        /// zero-friction path used while we focus on gameplay: it always yields a
        /// stable PlayerId the rest of the game can hang data off.
        /// </summary>
        Task<ServiceResult<PlayerAccount>> SignInAnonymouslyAsync();

        /// <summary>Create a recoverable Unity Authentication account.</summary>
        Task<ServiceResult<PlayerAccount>> SignUpAsync(string username, string password);

        /// <summary>Sign in to an existing Unity Authentication account.</summary>
        Task<ServiceResult<PlayerAccount>> SignInAsync(string username, string password);

        /// <summary>Attach credentials to the current anonymous account without changing its ID.</summary>
        Task<ServiceResult<PlayerAccount>> LinkUsernamePasswordAsync(string username, string password);

        /// <summary>
        /// Change the display name on the current account.
        /// </summary>
        Task<ServiceResult> SetDisplayNameAsync(string displayName);

        /// <summary>
        /// Clear the current session. Local id (if any) is kept so the next anonymous
        /// sign-in returns the same PlayerId.
        /// </summary>
        Task<ServiceResult> SignOutAsync();
    }
}
