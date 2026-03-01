Shader "Custom/DoodleDistort"
{
    Properties
    {
        _MainTex ("Sprite Texture", 2D) = "white" {}
        _DistortionMagnitude ("Distortion Magnitude", Float) = 5.0
        _DistortionSpeed ("Distortion Speed", Float) = 1.0
        _UpdateInterval ("Distortion Update Interval (sec)", Float) = 0.1
    }
    SubShader
    {
        Tags { "RenderType" = "Transparent" "Queue" = "Transparent" }
        LOD 200
        Blend SrcAlpha OneMinusSrcAlpha
        Cull Off
        ZWrite Off

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            float4 _MainTex_ST;

            float _DistortionMagnitude;
            float _DistortionSpeed;
            float _UpdateInterval;

            struct appdata_t
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };

            float2 distortUV(float2 uv, float time)
            {
                float angle = sin(uv.y * 10.0 + time) + cos(uv.x * 15.0 + time * 1.3);
                angle += sin((uv.x + uv.y) * 30.0 + time * 2.0);
                float2 offset = float2(cos(angle), sin(angle));
                return uv + offset * (_DistortionMagnitude * 0.001);
            }

            v2f vert(appdata_t v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                return o;
            }

            float sampleAlpha(float2 uv)
            {
                return tex2D(_MainTex, uv).a;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                float snappedTime = floor(_Time.y / _UpdateInterval) * _UpdateInterval;
                float time = snappedTime * _DistortionSpeed;
                
                float2 uv = distortUV(i.uv, time);
                fixed4 col = tex2D(_MainTex, uv);

                return col;
            }
            ENDCG
        }
    }
}
