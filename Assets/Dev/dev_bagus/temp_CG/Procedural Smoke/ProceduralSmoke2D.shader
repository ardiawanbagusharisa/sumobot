Shader "Custom/ProceduralSmoke2D"
{
    Properties
    {
        _MainTex ("Smoke Texture", 2D) = "white" {}
        _StartColor ("Start Color", Color) = (0.5,0.5,0.5,0.8)
        _EndColor ("End Color", Color) = (0.5,0.5,0.5,0)
        _RotationSpeed ("Rotation Speed", Float) = 30
        _Lifetime ("Lifetime", Float) = 2
        _StartTime ("Start Time", Float) = 0
        _Scale ("Scale", Float) = 1
    }
    SubShader
    {
        Tags { "RenderType"="Transparent" "Queue"="Transparent" }
        Blend SrcAlpha OneMinusSrcAlpha
        ZWrite Off
        Cull Off
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;
            float4 _StartColor;
            float4 _EndColor;
            float _RotationSpeed;
            float _Lifetime;
            float _StartTime;
            float _Scale;

            float _TimeY; // Unity time (shader-specific, auto-injected)

            v2f vert (appdata v)
            {
                v2f o;
                float time = _TimeY - _StartTime;
                float lifeT = saturate(time / _Lifetime);

                // Apply rotation
                float angle = radians(_RotationSpeed * time);
                float2 uv = v.uv - 0.5;
                float cosA = cos(angle);
                float sinA = sin(angle);
                float2 rotatedUV = float2(
                    uv.x * cosA - uv.y * sinA,
                    uv.x * sinA + uv.y * cosA
                ) + 0.5;

                o.uv = rotatedUV;
                o.vertex = UnityObjectToClipPos(v.vertex);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                float time = _TimeY - _StartTime;
                float lifeT = saturate(time / _Lifetime);

                fixed4 texColor = tex2D(_MainTex, i.uv);
                fixed4 color = lerp(_StartColor, _EndColor, lifeT);
                return texColor * color;
            }
            ENDCG
        }
    }
}
