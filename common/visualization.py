"""Build CSV, PNG and HTML from checkpoints"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def save_figure(fig, path):
    """Save a figure"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_label(metadata):
    """Build readable label for figures"""
    names = {"ce": "CE", "soft": "Soft Labels", "hxe": "HXE"}
    label = names[metadata["variant"]]
    if metadata["parameter"] is not None:
        symbol = "β" if metadata["variant"] == "soft" else "α"
        label += f" {symbol}={metadata['parameter']:g}"
    return f"{label} - {metadata['stage']} - seed {metadata['seed']}"


def plot_learning_curves(run_dir, label):
    """Plot loss, accuracy and top-5 train/validation."""
    history_path = run_dir / "history.csv"
    if not history_path.exists():
        return
    history = pd.read_csv(history_path)
    for metric, title in (("loss", "Loss"), ("accuracy", "Species accuracy"), ("top5_accuracy", "Top-5 accuracy")):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(history.epoch, history[metric], label="Train")
        ax.plot(history.epoch, history[f"val_{metric}"], "--", label="Validation")
        ax.set(title=label, xlabel="Epoch", ylabel=title)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=.2)
        ax.legend()
        save_figure(fig, run_dir / "figures" / f"learning_{metric}.png")


def plot_per_class(run_dir, split, label):
    """Print worst F1 and confusion per class"""
    table = pd.read_csv(run_dir / f"per_class_{split}.csv", dtype={"species": str})
    worst = table.sort_values("f1").head(25)
    fig, ax = plt.subplots(figsize=(9, max(4, len(worst) * .26)))
    ax.barh(worst.species if "species_name" in worst else worst.species, worst.f1)
    ax.invert_yaxis()
    ax.set(xlabel="F1", xlim=(0, 1), title=f"{label} - {split} - lowest F1")
    ax.grid(axis="x", alpha=.2)
    save_figure(fig, run_dir / "figures" / f"per_class_{split}.png")
    if len(table) <= 40:
        matrix = pd.read_csv(run_dir / f"confusion_{split}.csv", index_col=0)
        fig, ax = plt.subplots(figsize=(9, 8))
        image = ax.imshow(matrix.to_numpy(), cmap="Blues")
        ax.set_xticks(range(len(matrix)), matrix.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(matrix)), matrix.index, fontsize=7)
        ax.set(xlabel="Predicted species", ylabel="True species", title=f"{label} - {split}")
        fig.colorbar(image, ax=ax, label="Images")
        save_figure(fig, run_dir / "figures" / f"confusion_{split}.png")


def plot_dataset_counts(path, output_dir):
    """Plot repartition of images through species and splits"""
    if not path.exists():
        return
    table = pd.read_csv(path, dtype={"species": str})
    table = table.sort_values("train", ascending=False)
    large = len(table) > 40
    fig, ax = plt.subplots(figsize=(12 if large else max(10, len(table) * .15), 5))
    positions = np.arange(len(table)) if large else table.species
    bottom = np.zeros(len(table))
    for split in ("train", "validation", "test"):
        ax.bar(positions, table[split], bottom=bottom, label=split)
        bottom += table[split].to_numpy()
    if large :
        ax.set_xlabel("Species index, sorted by training frequency")
    else :
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.set(ylabel="Images", title="Dataset distribution by species")
    ax.legend()
    save_figure(fig, output_dir / "dataset_distribution.png")
    for rank in ("kingdom", "class", "order"):
        if rank not in table:
            continue

        column = f"{rank}_name" if f"{rank}_name" in table else rank
        grouped = (
            table.groupby(column)[["train", "validation", "test"]]
            .sum()
            .sort_values("train", ascending=False)
        )
        fig, ax = plt.subplots(figsize=(12, max(4, len(grouped) * .25)))
        bottom = np.zeros(len(grouped))

        for split in ("train", "validation", "test"):
            ax.barh(grouped.index, grouped[split], left=bottom, label=split)
            bottom += grouped[split].to_numpy()

        ax.invert_yaxis()
        ax.set(xlabel="Images", title=f"Dataset distribution by {rank}")
        ax.legend()
        save_figure(fig, output_dir / f"dataset_distribution_{rank}.png")


def plot_augmentation_preview(image_path, transforms, settings, output_path):
    """Compare different augmention fonr an image"""
    from PIL import Image
    with Image.open(image_path) as image:
        original = image.convert("RGB")
    mean = np.array(settings["mean"]).reshape(1, 1, 3)
    std = np.array(settings["std"]).reshape(1, 1, 3)
    fig, axes = plt.subplots(len(transforms), 5, figsize=(14, 3 * len(transforms)), squeeze=False)
    for row, (name, transform) in enumerate(transforms.items()):
        axes[row, 0].imshow(original)
        axes[row, 0].set_title(f"{name}\nOriginal")
        for col in range(1, 5):
            tensor = transform(original.copy())
            visible = np.clip(tensor.permute(1, 2, 0).numpy() * std + mean, 0, 1)
            axes[row, col].imshow(visible)
            axes[row, col].set_title(f"Example {col}")
    for ax in axes.flat:
        ax.axis("off")
    save_figure(fig, output_path)


def build_html(records, overall, seed_summary, output_path):
    """Build HTML report with Plotly"""
    def rows(frame):
        return json.loads(frame.to_json(orient="records", double_precision=15))

    runs = []
    for record in records:
        classes = {}
        for split, table in record["classes"].items():
            columns = [name for name in ("species", "species_name", "precision",
                       "recall", "f1", "support") if name in table]
            classes[split] = rows(table[columns])
        runs.append({"label": record["label"],
                     "history": rows(record["history"]) if record["history"] is not None else [],
                     "classes": classes})
    payload = {"runs": runs, "scores": rows(overall), "summary": rows(seed_summary)}
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    campaign = str(overall.campaign.iloc[0]) if "campaign" in overall and len(overall) else "Campagne"

    page = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ · résultats</title>
<style>
:root{--ink:#172d3d;--muted:#5d7080;--line:#dce5ea;--accent:#007d79}
*{box-sizing:border-box}body{margin:0;background:#f4f7f9;color:var(--ink);font:15px/1.5 system-ui,sans-serif}
main{max-width:1250px;margin:auto;padding:30px 22px 60px}
h1{font-size:30px;line-height:1.2;margin:8px 0;overflow-wrap:anywhere}h2{font-size:21px;margin:0 0 8px}
h3{font-size:16px;margin:0 0 8px}p{margin:6px 0 14px;color:var(--muted)}
.eyebrow{font-size:12px;letter-spacing:.14em;font-weight:750;color:var(--accent)}
nav{display:flex;gap:16px;flex-wrap:wrap;margin:18px 0 25px}nav a{color:var(--accent);text-decoration:none}
.card{background:white;border:1px solid var(--line);border-radius:14px;padding:22px;margin-bottom:20px}
.controls{display:flex;gap:16px;flex-wrap:wrap;align-items:end;margin:14px 0}
label{display:flex;flex-direction:column;gap:5px;font-size:13px;font-weight:650}
select,input{font:inherit;border:1px solid #bccbd4;border-radius:7px;padding:9px;background:white;color:var(--ink);max-width:100%}
#run{width:440px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}.grid>div{min-width:0}
.plot{width:100%;height:390px}.learning{height:auto;aspect-ratio:9/4;min-height:260px;max-height:470px}
#classes{height:640px}#distribution{height:320px}
.note{font-size:13px;background:#eef7f6;padding:12px 15px;border-radius:8px}
.stats{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0}.stat{background:#f4f7f9;border-radius:7px;padding:8px 12px}
.table-wrap{overflow:auto;max-height:460px;border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{position:sticky;top:0;background:#edf3f6;z-index:1}th:first-child,td:first-child{text-align:left}
tbody tr:hover{background:#f1faf9}.small{font-size:12px;color:var(--muted)}
@media(max-width:760px){main{padding:18px 12px}.card{padding:16px}.grid{grid-template-columns:1fr}
.plot{height:330px}.learning{height:auto;min-height:270px}#run{width:100%}.controls>label{max-width:100%}}
</style><script>__PLOTLY__</script></head><body><main>
<header><div class="eyebrow">CLASSIFICATION HIÉRARCHIQUE · RAPPORT HORS LIGNE</div>
<h1>__TITLE__</h1><p id="context"></p>
<nav><a href="#learning-section">Courbes</a><a href="#comparison-section">Comparaisons</a>
<a href="#species-section">Espèces</a></nav></header>
<section class="card" id="learning-section"><h2>Learning curves</h2>
<p>Un run à la fois : les échelles des losses dépendent de la méthode et de son paramètre.</p>
<div class="controls"><label>Run<select id="run"></select></label>
<label>Split d'évaluation<select id="split"></select></label></div>
<div class="note" id="run-note"></div>
<h3 style="margin-top:20px">Loss</h3><div id="loss" class="plot learning"></div>
<div class="grid"><div><h3>Accuracy top-1</h3><div id="accuracy" class="plot learning"></div></div>
<div><h3>Accuracy top-5</h3><div id="top5" class="plot learning"></div></div></div>
<p class="small">Valeurs originales des historiques, sans lissage. Trait vertical : meilleur checkpoint selon la loss de validation.
Les métriques finales ci-dessous correspondent à ce checkpoint, pas forcément à la dernière epoch.</p></section>
<section class="card" id="comparison-section"><h2>Comparaison des méthodes</h2>
<p>Les comparaisons utilisent le split sélectionné. Couleur = méthode ; survol = run.</p>
<div class="grid"><div><h3>Erreur et sévérité</h3><div id="tradeoff" class="plot"></div></div>
<div><h3>Accuracy by taxonomic rank · run sélectionné</h3><div id="ranks" class="plot"></div></div></div>
<p class="small">Sévérité : distance moyenne parmi les erreurs. Distance @1 : moyenne sur toutes les images.
Dans le graphique de gauche, les valeurs faibles sur les deux axes sont préférables.</p>
<div id="metrics" class="table-wrap"></div><h3 style="margin-top:24px">Moyenne et écart-type entre seeds</h3>
<p class="small">Avec une seule seed, l'écart-type est indiqué par « — ».</p><div id="summary" class="table-wrap"></div></section>
<section class="card" id="species-section"><h2>F1 by species</h2>
<p>Analyse du run et du split sélectionnés. Rechercher un identifiant natXXXX ou un nom scientifique.</p>
<div class="controls"><label>Classement<select id="order">
<option value="worst">F1 les plus faibles</option><option value="best">F1 les plus élevés</option></select></label>
<label>Recherche<input id="search" type="search" placeholder="nat0014, nom d'espèce…"></label></div>
<div id="class-stats" class="stats"></div>
<div class="grid"><div><h3>25 espèces du classement</h3><div id="classes" class="plot"></div></div>
<div><h3>Distribution des F1</h3><div id="distribution" class="plot"></div>
<p class="small">Distribution sur toutes les espèces. Les points à F1 = 0 sont affichés explicitement.</p></div></div>
<p id="class-count" class="small"></p><div id="class-table" class="table-wrap"></div></section>
<footer class="small">Rapport autonome : aucune connexion internet nécessaire. Export PNG disponible dans la barre de chaque graphique.
Les CSV complets restent disponibles dans les dossiers runs/ et reports/.</footer>
</main><script id="report-data" type="application/json">__DATA__</script><script>
const report=JSON.parse(document.getElementById('report-data').textContent);
const el=id=>document.getElementById(id);
const colors={ce:'#246eb9',soft:'#e28b20',hxe:'#007d79'};
const esc=value=>String(value??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=(v,percent=false)=>v==null||!Number.isFinite(Number(v))?'—':percent?(100*v).toFixed(2)+' %':Number(v).toFixed(3);
const config={responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',scale:2}};
function draw(id,traces,extra={}){
    return Plotly.react(id,traces,{autosize:true,paper_bgcolor:'white',plot_bgcolor:'white',
        font:{family:'system-ui, sans-serif',size:12,color:'#172d3d'},margin:{l:60,r:20,t:30,b:65},
        xaxis:{gridcolor:'#e8eef2',zeroline:false,automargin:true},
        yaxis:{gridcolor:'#e8eef2',zeroline:false,automargin:true},
        legend:{orientation:'h',x:0,y:1.15},hovermode:'closest',...extra},config);
}
function table(id,headers,values){
    el(id).innerHTML='<table><thead><tr>'+headers.map(h=>'<th>'+esc(h)+'</th>').join('')+
        '</tr></thead><tbody>'+values.map(row=>'<tr>'+row.map(v=>'<td>'+esc(v)+'</td>').join('')+'</tr>').join('')+
        '</tbody></table>';
}
function selectedRun(){return report.runs[Number(el('run').value)];}
function renderLearning(){
    const run=selectedRun(), history=run?.history??[];
    const best=history.filter(r=>r.val_loss!=null).reduce((a,b)=>!a||b.val_loss<a.val_loss?b:a,null);
    el('run-note').textContent=best?'Meilleur checkpoint : epoch '+best.epoch+
        ' / '+history.length+' · loss validation '+fmt(best.val_loss):'Historique non disponible.';
    for(const [id,column,percent] of [['loss','loss',false],['accuracy','accuracy',true],['top5','top5_accuracy',true]]){
        const values=history.flatMap(r=>[r[column],r['val_'+column]]).filter(v=>Number.isFinite(v));
        const low=values.length?Math.min(...values):0, high=values.length?Math.max(...values):1;
        const pad=Math.max((high-low)*.05,Math.abs(high)*.005,1e-6);
        const traces=[['Train',column,'#246eb9','solid'],['Validation','val_'+column,'#e28b20','dash']].map(([name,key,color,dash])=>
            ({x:history.map(r=>r.epoch),y:history.map(r=>r[key]),name,type:'scatter',mode:'lines',
              line:{color,dash,width:2.5},hovertemplate:percent?'Epoch %{x}<br>%{y:.2%}<extra>%{fullData.name}</extra>':
              'Epoch %{x}<br>%{y:.5f}<extra>%{fullData.name}</extra>'}));
        draw(id,traces,{xaxis:{title:'Epoch',dtick:history.length<=12?1:undefined,gridcolor:'#e8eef2',automargin:true},
            yaxis:{title:percent?'Accuracy':'Loss',tickformat:percent?'.0%':undefined,range:[low-pad,high+pad],gridcolor:'#e8eef2'},
            shapes:best?[{type:'line',x0:best.epoch,x1:best.epoch,y0:0,y1:1,yref:'paper',
                line:{color:'#9baab5',dash:'dot',width:1}}]:[]});
    }
}
function renderComparison(){
    const scores=report.scores.filter(r=>r.split===el('split').value);
    const variants=[...new Set(scores.map(r=>r.variant))];
    draw('tradeoff',variants.map(variant=>{
        const rows=scores.filter(r=>r.variant===variant);
        return {type:'scatter',mode:'markers',name:variant.toUpperCase(),x:rows.map(r=>r.top1_error),
            y:rows.map(r=>r.mistake_severity),text:rows.map(r=>r.label),marker:{size:11,color:colors[variant]},
            hovertemplate:'%{text}<br>Erreur %{x:.2%}<br>Sévérité %{y:.3f}<extra></extra>'};
    }),{xaxis:{title:'Erreur top-1',tickformat:'.0%',gridcolor:'#e8eef2'},
        yaxis:{title:'Sévérité des erreurs',gridcolor:'#e8eef2'}});
    const run=selectedRun(), row=scores.find(r=>r.label===run?.label);
    const ranks=['kingdom','phylum','class','order','family','genus','species'].filter(rank=>row?.[rank+'_accuracy']!=null);
    draw('ranks',row?[{type:'scatter',mode:'lines+markers',x:ranks,y:ranks.map(rank=>row[rank+'_accuracy']),
        name:row.label,line:{color:colors[row.variant]??'#246eb9'},hovertemplate:'%{x}<br>%{y:.2%}<extra></extra>'}]:[],
        {showlegend:false,yaxis:{title:'Accuracy',range:[0,1],tickformat:'.0%',gridcolor:'#e8eef2'}});
    table('metrics',['Run','Top-1','Top-5','F1 macro','Sévérité','Distance @1'],scores.map(r=>
        [r.label,fmt(r.top1_accuracy,true),fmt(r.top5_accuracy,true),fmt(r.f1_macro,true),
         fmt(r.mistake_severity),fmt(r['avg_hierarchical_distance@1'])]));
    const summary=report.summary.filter(r=>r.split===el('split').value);
    const stat=(r,key,percent=false)=>fmt(r[key+'_mean'],percent)+' ± '+fmt(r[key+'_std'],percent);
    table('summary',['Variante / phase','Seeds','Top-1','F1 macro','Sévérité'],summary.map(r=>
        [r.variant.toUpperCase()+(r.parameter==null?'':' '+r.parameter)+' · '+r.stage,r.n_seeds,
         stat(r,'top1_accuracy',true),stat(r,'f1_macro',true),stat(r,'mistake_severity')]));
}
function renderClasses(){
    const all=selectedRun()?.classes[el('split').value]??[], query=el('search').value.trim().toLowerCase();
    const sorted=all.filter(r=>(r.species+' '+(r.species_name??'')).toLowerCase().includes(query))
        .sort((a,b)=>(el('order').value==='worst'?a.f1-b.f1:b.f1-a.f1)||a.species.localeCompare(b.species));
    const top=sorted.slice(0,25), names=top.map(r=>r.species_name?r.species+' · '+r.species_name:r.species);
    const labels=names.map(name=>name.length>34?name.slice(0,32)+'…':name);
    draw('classes',[{type:'bar',orientation:'h',x:top.map(r=>r.f1),y:labels,customdata:top.map((r,i)=>[names[i],r.support]),
        marker:{color:'#007d79'},hovertemplate:'%{customdata[0]}<br>F1 %{x:.2%}<br>Support %{customdata[1]}<extra></extra>'},
        {type:'scatter',mode:'markers+text',x:top.map(r=>r.f1),y:labels,text:top.map(r=>fmt(r.f1,true)),
         textposition:'middle right',marker:{size:5,color:'#007d79'},hoverinfo:'skip'}],
        {showlegend:false,margin:{l:160,r:50,t:15,b:50},xaxis:{title:'F1',range:[-.02,1.15],tickvals:[0,.25,.5,.75,1],tickformat:'.0%',gridcolor:'#e8eef2'},
         yaxis:{autorange:'reversed',automargin:true}});
    const bins=Array(20).fill(0);
    for(const row of all) bins[Math.min(19,Math.floor(row.f1*20))]++;
    draw('distribution',[{type:'bar',x:bins.map((_,i)=>(i+.5)/20),y:bins,width:.048,
        customdata:bins.map((_,i)=>[i/20,(i+1)/20]),marker:{color:'#246eb9'},
        hovertemplate:'F1 %{customdata[0]:.0%}–%{customdata[1]:.0%}<br>%{y} espèces<extra></extra>'}],
        {showlegend:false,xaxis:{title:'F1',range:[0,1],tickformat:'.0%'},yaxis:{title:'Espèces',gridcolor:'#e8eef2'}});
    el('class-stats').innerHTML=['Espèces : '+all.length,'F1 nul : '+all.filter(r=>r.f1===0).length,
        'F1 macro : '+fmt(all.length?all.reduce((sum,r)=>sum+r.f1,0)/all.length:null,true)]
        .map(text=>'<span class="stat">'+esc(text)+'</span>').join('');
    el('class-count').textContent=Math.min(sorted.length,50)+' lignes affichées sur '+sorted.length+
        '. La recherche parcourt toutes les espèces ; le CSV conserve la liste complète.';
    table('class-table',['Espèce','Nom scientifique','Support','Précision','Rappel','F1'],sorted.slice(0,50).map(r=>
        [r.species,r.species_name,r.support,fmt(r.precision,true),fmt(r.recall,true),fmt(r.f1,true)]));
}
function refresh(){renderLearning();renderComparison();renderClasses();}
el('run').innerHTML=report.runs.map((run,i)=>'<option value="'+i+'">'+esc(run.label)+'</option>').join('');
const splits=[...new Set(report.scores.map(r=>r.split))];
el('split').innerHTML=(splits.length?splits:['validation']).map(split=>'<option>'+esc(split)+'</option>').join('');
el('split').value=splits.includes('validation')?'validation':splits[0]??'validation';
const first=report.scores[0], seeds=new Set(report.scores.map(r=>r.seed));
el('context').textContent=report.runs.length+' runs · '+seeds.size+' seeds'+
    (first?' · '+first.model_key+' · '+first.augmentation:'');
el('run').addEventListener('change',refresh);el('split').addEventListener('change',()=>{renderComparison();renderClasses();});
el('order').addEventListener('change',renderClasses);el('search').addEventListener('input',renderClasses);
refresh();
</script></body></html>"""
    page = page.replace("__TITLE__", escape(campaign)).replace("__DATA__", data)
    page = page.replace("__PLOTLY__", get_plotlyjs())
    Path(output_path).write_text(page, encoding="utf-8")


def build_reports(campaign_dir):
    """Build graphics with available with available plots"""
    campaign_dir = Path(campaign_dir)
    if not campaign_dir.is_dir():
        raise FileNotFoundError(campaign_dir)
    reports_dir = campaign_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    records, rows, class_tables = [], [], []
    stage_order = {"frozen": 0, "partial": 1, "deeper": 2}
    paths = sorted((campaign_dir / "runs").glob("*/*/metadata.json"),
                   key=lambda path: (path.parent.parent.name, stage_order[path.parent.name]))
    for path in paths:
        run_dir = path.parent
        metadata = json.loads(path.read_text(encoding="utf-8"))
        label = run_label(metadata)
        plot_learning_curves(run_dir, label)
        history_path = run_dir / "history.csv"
        record = {"label": label, "history": pd.read_csv(history_path) if history_path.exists() else None, "classes": {}}
        for split in ("validation", "test"):
            metrics_path = run_dir / f"metrics_{split}.json"
            if not metrics_path.exists():
                continue
            scores = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append({**{key: value for key, value in metadata.items() if key != "training"},
                         "label": label, "split": split, **scores})
            record["classes"][split] = pd.read_csv(run_dir / f"per_class_{split}.csv", dtype={"species": str})
            table = record["classes"][split].copy()
            for key, value in (("label", label), ("seed", metadata["seed"]), ("stage", metadata["stage"]),
                               ("variant", metadata["variant"]), ("parameter", metadata["parameter"]), ("split", split)):
                table[key] = np.nan if key == "parameter" and value is None else value
            class_tables.append(table)
            plot_per_class(run_dir, split, label)
        records.append(record)
    overall = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["label", "split"])
    overall.to_csv(reports_dir / "comparison.csv", index=False)
    if class_tables:
        pd.concat(class_tables, ignore_index=True).to_csv(reports_dir / "per_class_comparison.csv", index=False)
    summary = pd.DataFrame()
    if rows:
        groups = ["campaign", "model_key", "augmentation", "variant", "parameter", "stage", "split"]
        metrics = [column for column in overall.select_dtypes(include="number") if column not in {"seed", "parameter"}]
        grouped = overall.groupby(groups, dropna=False)
        summary = grouped[metrics].agg(["mean", "std"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary["n_seeds"] = grouped.seed.nunique()
        summary = summary.reset_index()
        for split in overall.split.unique():
            table = overall[overall.split == split]
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].bar(range(len(table)), table.top1_accuracy)
            axes[0].set_xticks(range(len(table)), table.label, rotation=90, fontsize=7)
            axes[0].set(title=f"Species accuracy - {split}", ylim=(0, 1))
            for variant in ("ce", "soft", "hxe"):
                selected = table[table.variant == variant]
                if not selected.empty:
                    axes[1].scatter(selected.top1_error, selected.mistake_severity, label=variant)
            axes[1].set(xlabel="Top-1 error", ylabel="Mistake severity", title="Hierarchical trade-off")
            axes[1].legend()
            axes[1].grid(alpha=.2)
            save_figure(fig, reports_dir / f"comparison_{split}.png")
            for metric in ("avg_hierarchical_distance@1", "avg_hierarchical_distance@5"):
                fig, ax = plt.subplots(figsize=(8, 5))
                for variant in ("ce", "soft", "hxe"):
                    selected = table[table.variant == variant]
                    if not selected.empty:
                        ax.scatter(selected.top1_error, selected[metric], label=variant)
                ax.set(xlabel="Top-1 error", ylabel=metric, title=f"Hierarchical distance - {split}")
                ax.legend()
                ax.grid(alpha=.2)
                save_figure(fig, reports_dir / f"{metric.replace('@', '_')}_{split}.png")
    summary.to_csv(reports_dir / "seed_summary.csv", index=False)
    plot_dataset_counts(reports_dir / "dataset_counts.csv", reports_dir)
    build_html(records, overall, summary, reports_dir / "report.html")
    print("Rapport :", reports_dir / "report.html")
    return overall


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path, help="File containing runs/ and campaign.json")
    build_reports(parser.parse_args().campaign_dir)
