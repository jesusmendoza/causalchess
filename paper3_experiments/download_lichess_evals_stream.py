import requests
import zstandard as zstd
import json
import sys
import io

URL = "https://database.lichess.org/lichess_db_eval.jsonl.zst"
OUTPUT_FILE = "lichess_15m_evals.tsv"
TARGET_POSITIONS = 15_000_000

def main():
    print(f"Iniciando descarga ultra-segura O(1) de {URL}...")
    print(f"Objetivo: {TARGET_POSITIONS} posiciones. La RAM se mantendrá cerca de 0.")
    
    response = requests.get(URL, stream=True)
    response.raise_for_status()
    
    dctx = zstd.ZstdDecompressor()
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        out_f.write("fen\tscore\n")
        count = 0
        
        # Envolvemos el stream en un TextIOWrapper para manejo de memoria O(1) puro
        with dctx.stream_reader(response.raw) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            for line in text_stream:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    if "evals" in data and len(data["evals"]) > 0:
                        evals = data["evals"][0]
                        if "pvs" in evals and len(evals["pvs"]) > 0:
                            cp = evals["pvs"][0].get("cp")
                            if cp is not None:
                                fen = data["fen"]
                                out_f.write(f"{fen}\t{cp}\n")
                                count += 1
                                
                                if count % 100000 == 0:
                                    # flush=True fuerza la liberación de buffers de disco
                                    print(f"Progreso: {count} / {TARGET_POSITIONS} extraídas...", flush=True)
                                    out_f.flush()
                                    
                                if count >= TARGET_POSITIONS:
                                    break
                except Exception:
                    pass
                        
    print(f"\n¡Extracción completada! Se guardaron {count} posiciones en {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
