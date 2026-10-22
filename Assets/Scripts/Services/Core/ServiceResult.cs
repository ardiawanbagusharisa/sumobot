using System;

namespace SumoServices
{
    /// <summary>
    /// Uniform result wrapper for every service call. Because the real backends
    /// (Photon / UGS / any cloud) are asynchronous and can fail on the network,
    /// services never throw for expected failures — they return a ServiceResult
    /// so callers can branch on Success without try/catch everywhere.
    /// </summary>
    public readonly struct ServiceResult
    {
        public bool Success { get; }
        public string Error { get; }

        private ServiceResult(bool success, string error)
        {
            Success = success;
            Error = error;
        }

        public static ServiceResult Ok() => new(true, null);
        public static ServiceResult Fail(string error) => new(false, error);
    }

    /// <summary>
    /// Result wrapper that also carries a payload on success.
    /// </summary>
    public readonly struct ServiceResult<T>
    {
        public bool Success { get; }
        public string Error { get; }
        public T Value { get; }

        private ServiceResult(bool success, T value, string error)
        {
            Success = success;
            Value = value;
            Error = error;
        }

        public static ServiceResult<T> Ok(T value) => new(true, value, null);
        public static ServiceResult<T> Fail(string error) => new(false, default, error);
    }
}
