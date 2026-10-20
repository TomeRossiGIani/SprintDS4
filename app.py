"""Protótipo HTTP local sem dependências de frontend ou serviços externos."""

import json, argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from pipeline import Analyzer

HTML = """<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Kaivi AI | Inteligência comercial</title><style>
:root{font-family:system-ui;color:#172b42;background:#edf2f7}body{max-width:1100px;margin:40px auto;padding:0 22px}h1{font-size:42px;margin-bottom:4px}p{line-height:1.6}.pill{background:#dbe9ff;padding:6px 12px;border-radius:20px;font-size:13px}textarea{box-sizing:border-box;width:100%;height:210px;padding:18px;border:1px solid #9baec5;border-radius:12px;font:16px system-ui}.panel{background:white;border-radius:18px;padding:26px;margin:22px 0;box-shadow:0 5px 20px #1232}button{background:#1754c7;color:white;border:0;border-radius:9px;padding:13px 18px;cursor:pointer;font-size:15px;margin:10px 6px 0 0}button:disabled{opacity:.5}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px}.card{padding:18px;border:1px solid #d8e3ef;border-radius:12px}.number{font-size:32px;color:#1754c7;font-weight:700}li{margin:12px 0;line-height:1.5}blockquote{border-left:3px solid #94add1;padding-left:14px;margin-left:0}.warning{color:#7b4400;background:#fff4db;padding:14px;border-radius:9px}small{color:#536780}blockquote small{display:block;margin-top:7px}#status{min-height:24px}input{margin:12px 0}</style>
<span class="pill">DATA SCIENCE · SPRINT 4</span><h1>Kaivi AI</h1><p>Transforme a transcrição em prioridades para sua equipe comercial.</p>
<div class="warning">Demonstração acadêmica com modelos treinados em dados sintéticos. Revise os resultados antes de agir. Os scores não representam probabilidades reais calibradas.</div>
<section class="panel"><h2>Analisar reunião</h2><p>Cole uma transcrição em português ou carregue um arquivo TXT UTF-8. Limite: 50.000 caracteres. O processamento é local.</p><label for="text">Transcrição</label><textarea id="text" placeholder="Cliente: Utilizamos o Protheus..."></textarea><input id="file" type="file" accept=".txt,text/plain" aria-label="Carregar transcrição TXT"><br><button id="analyze">Gerar inteligência</button><button id="example">Carregar exemplo</button><button id="download" hidden>Baixar resultado JSON</button><p id="status" role="status"></p></section><section id="output" hidden></section>
<script>
const $=id=>document.getElementById(id); let last=null;
function node(tag,text,parent){const n=document.createElement(tag);n.textContent=text;parent.appendChild(n);return n}
$('example').onclick=()=>{$('text').value='Cliente: Estamos insatisfeitos com o suporte do Protheus e podemos cancelar o contrato. Também precisamos de uma proposta para novas licenças.'};
$('file').onchange=async e=>{const f=e.target.files[0];if(f){if(f.size>200000){$('status').textContent='Arquivo muito grande.';return}$('text').value=await f.text()}};
$('analyze').onclick=async()=>{ $('analyze').disabled=true;$('status').textContent='Processando…';$('output').hidden=true;$('download').hidden=true;try{const response=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({texto:$('text').value})});const data=await response.json();if(!response.ok)throw Error(data.erro);last=data;const out=$('output');out.replaceChildren();out.className='panel';out.hidden=false;node('h2','Inteligência da reunião',out);node('p','Produtos mencionados: '+data.produtos.join(', '),out);const grid=node('div','',out);grid.className='grid';Object.entries(data.scores).forEach(([label,value])=>{let card=node('div','',grid);card.className='card';node('div',label.replaceAll('_',' '),card);node('div',value+' / 100',card).className='number'});node('p',Object.entries(data.classificacoes).map(([k,v])=>k+': '+v.replaceAll('_',' ')).join(' · '),out);node('p','Confiança mínima: '+data.confianca_minima+' | Cobertura de vocabulário: '+data.cobertura_vocabulario,out);if(data.revisao_humana)node('p','Validação humana obrigatória: modelos demonstrativos treinados em dados sintéticos.',out).className='warning';node('h3','Próximas ações',out);const ul=node('ul','',out);data.recomendacoes.forEach(x=>node('li',x,ul));node('h3','Termos encontrados',out);node('p',data.termos.join(' · ')||'Nenhum termo conhecido.',out);node('p',data.falantes==='cliente_identificado'?'Classificações calculadas sobre as falas marcadas como Cliente ou Comprador.':'Texto sem identificação de falantes: confirme quem declarou cada intenção.',out);node('h3','Trechos para conferência',out);node('small','Trechos originais; não são explicações causais dos modelos.',out);data.evidencias.forEach(x=>{const q=node('blockquote',x.trecho,out);if(x.sinais)node('small','Sinal para conferir: '+x.sinais.join(', '),q)});$('download').hidden=false;$('status').textContent='Análise concluída.'}catch(e){$('status').textContent=e.message}finally{$('analyze').disabled=false}};
$('download').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(last,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='kaivi_resultado.json';a.click();URL.revokeObjectURL(url)};
</script></html>"""


class Handler(BaseHTTPRequestHandler):
    analyzer = None

    def send(self, status, data, mime="application/json; charset=utf-8"):
        body = (
            data.encode("utf-8")
            if isinstance(data, str)
            else json.dumps(data, ensure_ascii=False).encode("utf-8")
        )
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self.send(200, HTML, "text/html; charset=utf-8")
        elif self.path == "/health":
            self.send(200, {"status": "ok"})
        else:
            self.send(404, {"erro": "Rota inexistente."})

    def do_POST(self):
        if self.path != "/api/analyze":
            self.send(404, {"erro": "Rota inexistente."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 300000:
                self.send(413, {"erro": "Corpo ausente ou muito grande."})
                return
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Envie um objeto JSON com o campo texto.")
            self.send(200, self.analyzer.analyze(data.get("texto")))
        except (ValueError, UnicodeDecodeError):
            self.send(
                400,
                {
                    "erro": "Transcrição inválida: envie JSON com texto de 5 palavras a 50.000 caracteres."
                },
            )

    def log_message(self, *args):
        pass  # Não registrar transcrições no terminal.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    Handler.analyzer = Analyzer()
    print(f"Kaivi AI: http://127.0.0.1:{args.port} (Ctrl+C para sair)", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
