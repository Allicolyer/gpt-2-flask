# GPT-2 Flask App

`http://185.230.222.21:5000/generate` (for spring)

body for `POST` request

```json
{
    "context": "hello what a story I am telling! Do you like my story?",
    "config": {
        "length": int,
        "seed": int,
        "temperature": float,
        "repetition_penalty": float,
        "k": float,
        "p": float,
        "no_cuda": bool
    }
}
```
