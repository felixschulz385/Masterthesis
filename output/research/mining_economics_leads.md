# Gold, enforcement and downstream health: research leads

*Draft, 2026-10-22. Builds on the pollution and health arms of this project
(`output/notebooks/pollution/memo.md`, `output/notebooks/health/memo.md`).*

## Summary

The project's empirical arms show that upstream land-cover change moves river water
quality in a few specific ways, but that this mostly does not reach health:

- **Pollution.** Pasture raises lead and cadmium violations, agriculture raises COD, and
  mining raises mercury violations.
- **Health.** There is no detectable effect of upstream pasture or agriculture on
  hospitalizations, infant deaths or birth outcomes.
- **The exception is mining in the Amazon.** Where mining appears within 50 km upstream,
  infant deaths rise by about 23% (bootstrap p = 0.002), and the injury placebo does not
  move.
- **That result is thin.** It rests on 36 mining onsets in 18 basins, and there is a
  small bump two years before onset.

Mining is therefore where an economic story can carry the health evidence. This report
develops four connected leads:

1. **Gold-laundering policy:** laundering policy drives garimpo, and garimpo drives
   downstream contamination and health.
2. **A two-group model of garimpo:** outside crews move to where enforcement is weak, and
   local residents turn to mining as their traditional economy erodes.
3. **Enforcement displacement:** how much mining enforcement relocates rather than
   removes, and where the health burden goes.
4. **Indigenous-land demarcation:** whether the gap between legal protection and actual
   control attracts mining into protected territory.

Leads 1, 3 and 4 each move upstream mining for a different reason (price, enforcement
cost, legal status), and lead 2 predicts who responds to each. Together they give three
independent sources of variation for one downstream health effect, and a test of
mechanism. The proposed core paper combines leads 1–3; lead 4 is a section or a separate
paper.

## Why the project is well placed

- **Directed river network.** The network links every 25 km stretch of river to the
  monitoring stations and municipalities downstream of it. Few land-use papers can
  connect a mining site to the specific populations that drink and fish its water.
- **Pipeline already built.** Water quality (CONAMA violations including mercury),
  municipal health outcomes (SIH, SINASC, SIM) and inference with few clusters (Otto-3
  basins, wild-cluster bootstrap for OLS and PPML) are already in place.
- **Plugging in a design.** A new source of variation in upstream mining only needs to
  enter at the exposure stage; everything downstream is reused.

## Lead 1: gold-laundering policy → garimpo → downstream health

**Question.** Does making illegal gold easier or harder to sell change where garimpo
happens, and does that show up downstream in mercury violations and infant health?

**Mechanism.**

- Illegal gold enters the formal market through small-scale mining permits
  (*permissões de lavra garimpeira*) that declare sales, often without matching
  extraction.
- Law 12,844/2013 let gold buyers rely on the seller's own declaration of legal origin
  (the "good-faith" presumption), which lowered the cost of laundering.
- The Supreme Court suspended that presumption in 2023, and electronic invoices for gold
  followed, which raised the cost again. *Exact dates to be verified.*
- Laundering cost acts as a tax on the gold price miners receive. It should matter most
  for mining near permits that can absorb its output.

**Design.**

```
mining_jt = β1·(post-2013_t × ghost-permit exposure_j) + β2·(post-2023_t × ghost-permit exposure_j)
          + gold price_t × gold geology_j + site FE + state×year FE
```

- **Exposure.** Permit area near site *j* that declares gold sales but shows little
  mining footprint, measured before 2013 so the permits do not respond to the law.
- **Laundering measure.** Declared output (royalty payments) minus observed extraction
  (mapped mining area).
- **Downstream outcomes.** Mercury and lead violations at the stations below *j*, and
  infant deaths in the municipalities below *j*.
- **Estimation.** First the reduced form, then the law as an instrument for upstream
  mining.

**Data.**

- **In the pipeline:** the river network, water-quality stations, health outcomes, and a
  mining exposure.
- **To add:**
  - mining-permit polygons with type, mineral and dates (National Mining Agency,
    SIGMINE);
  - royalty payments by permit holder (CFEM);
  - the MapBiomas mining module, which separates garimpo from industrial mining and gold
    from other ores;
  - a gold-geology layer (Geological Survey of Brazil).

**Threats and checks.**

- **Permits proxy for geology.** Compare permits that declare sales without extracting
  against active permits, and control for gold price × geology.
- **2023 coincides with Lula's enforcement surge.** The surge targeted Indigenous lands
  and specific rivers, while the laundering change applied everywhere near permits, so
  the two can be separated in space.
- **Pre-trends and placebo.** Check for trends before 2013, and run a placebo with
  permits for minerals other than gold.

**Contribution.** Laundering is documented descriptively. This gives a causal estimate
of how laundering policy affects mining and, through rivers, health, aimed at the "gold
flows" layer that enforcement based on deforestation maps cannot see. It also adds many
more mining onsets than the current 36.

## Lead 2: a two-group model of garimpo supply

**Question.** Is garimpo driven by outside crews moving to where enforcement is weak, or
by locals turning to mining as their traditional economy erodes, and how do the two
interact?

**Model.**

- **Outside crews.** Mechanized crews (dredges, excavators) and their financiers are
  mobile and choose sites by expected profit:

  ```
  (gold price − laundering cost) × deposit quality − access cost − P(equipment destroyed) × equipment value
  ```

  Enforcement enters as a *relative* cost: concentrating it on one territory moves crews
  to the next-best deposit rather than out of mining.
- **Local residents.** Riverine, Indigenous and smallholder households choose between
  mining (working for crews, or deals with them) and the traditional economy: fishing,
  hunting, forest products, small-plot farming, transfers. They move into mining when
  that economy erodes, through fewer fish, deforestation, drought or lost cash income.
- **Feedback.** Crews contaminate and silt rivers, fishing declines downstream, and
  downstream residents move into mining. Mining can spread down a river without new
  deposits being found.
- **Policy implication.** Removing crews does not restore residents' livelihoods, which
  predicts quick relapse after enforcement.

**Predictions.**

1. **Displacement.** Outside crews respond to laundering and enforcement shocks, and are
   displaced rather than removed (tested in lead 3).
2. **Downstream asymmetry.** A mining onset raises later onsets downstream more than
   upstream at the same river distance. Deposits and crew movement are roughly
   symmetric along a river, while contamination flows only downstream.
3. **Fishing dependence.** Residents respond more where fishing matters and after
   droughts (e.g. the record-low rivers of 2023–24).
4. **Relapse.** Mining returns faster after enforcement where the traditional economy has
   already degraded.

**Data.**

- **Separating the two groups.** Malaria surveillance records (SIVEP-Malária) give the
  patient's occupation (including mining), likely place of infection and residence.
  Mining-related cases infected at a site but resident elsewhere measure outside crews;
  resident cases measure local participation. *Microdata access to be checked.*
- **Fishing dependence.** Registered artisanal fishers and closed-season payments to
  fishers (*seguro-defeso*), both public by municipality.
- **Drought.** ERA5 precipitation and soil moisture, already in the pipeline.
- **River direction.** The asymmetry test uses the project's directed reachability and
  distance matrices directly.

**Threats.**

- **Geology.** Alluvial gold concentrates in lower reaches, which confounds the asymmetry
  test; control for geology.
- **Malaria programmes.** Malaria cases also respond to control programmes; use shares
  and state×year fixed effects.
- **Dredges.** Dredge mining on rivers leaves little land footprint and is missed by
  MapBiomas.

**Contribution.** The most novel lead. The downstream-asymmetry test is only possible
with a directed river network linked to outcomes. The model also says which policy
works: if residents drive mining, enforcement alone relapses.

## Lead 3: enforcement displacement

**Question.** When enforcement shuts down mining in one territory, how much moves
elsewhere, where does it go, and does the health burden move with it?

**Shocks.**

- the Yanomami health emergency and enforcement operation (2023);
- large dredge operations on the Madeira River (2021, 2024, Operation Boiuna 2025);
- operations in Munduruku and Kayapó lands;
- in the other direction, the 2019 drop in enforcement credibility, including political
  statements against destroying mining equipment.

**Design.**

- **Substitute sites.** Defined in advance as gold-bearing areas with river access that
  were not targeted, ranked by travel cost from the targeted territory along rivers and
  roads.
- **Event study.** Mining onsets at substitute sites by distance band, around each
  shock.
- **Displacement ratio.** The increase at substitute sites divided by the decrease in
  the targeted area; a ratio near 1 means pure relocation.
- **Health.** Whether mercury violations and infant deaths rise below the substitute
  sites as they fall below the targeted one.

**Data.**

- **To add:** IBAMA fines, embargoes and seizures (geocoded, public), with operation
  dates from official reporting.
- **In the pipeline or planned:** the MapBiomas mining module and the river network for
  travel distances and downstream outcomes.

**Threats.**

- **Few operations.** Few large operations means few independent units; use
  randomization inference across placebo operation dates.
- **Coinciding national changes.** Several shocks coincide with national changes; use
  state×year fixed effects and placebo substitute sites without gold.
- **Dredges.** Dredge mining is invisible to land-cover maps; IBAMA seizure records partly
  fill the gap.

**Contribution.** It quantifies what is currently anecdotal (Yanomami → Kayapó, dredges
returning on the Madeira within months). A finding that enforcement relocates the health
burden rather than removing it would be a strong policy result, and it directly tests
prediction 1 of lead 2.

## Lead 4: Indigenous-land demarcation and mining inside protected territory

**Question.** Does the gap between legal protection and actual control, especially while
demarcation is pending, attract garimpo into Indigenous lands and protected areas?

**Mechanism.**

- **Protection without control.** Protected status blocks legal mining but not demand.
  Lands with intact deposits and weak state presence become targets.
- **Exposure while pending.** Demarcation of Indigenous lands moves through declared,
  delimited, homologated and regularized stages, and the pending stages may be the most
  exposed.
- **Anticipation.** Bills to legalize mining on Indigenous lands (e.g. PL 191/2020), and
  the mining-permit requests filed over them, act as signals that miners anticipate.

**Design.**

- **Demarcation.** A before-after comparison of mining inside Indigenous lands around the
  homologation date, exploiting the staggered stages.
- **Protected areas.** Strict-protection versus sustainable-use categories.
- **Anticipation.** An event study of permit requests and mining onsets around the
  introduction of PL 191/2020.
- **Downstream.** Effects on the municipalities below these lands.

**Data.**

- **To add:**
  - Indigenous-land boundaries with stage and decree dates (FUNAI);
  - protected-area boundaries with creation dates (national register of conservation
    units);
  - permit requests over Indigenous lands (SIGMINE).
- **In the pipeline:** mining and downstream outcomes.

**Threats.**

- **Political timing.** Demarcation timing is political and may itself respond to mining
  pressure; check for trends before demarcation.
- **Related literature.** Work on property rights in Indigenous lands and deforestation
  exists (e.g. Baragwanath & Bayi 2020), though not on mining.
- **Indigenous health data.** Indigenous health is poorly captured in municipal DATASUS
  data; it is recorded in a separate system run by the indigenous health secretariat
  (SESAI).

**Contribution.** It explains where displaced crews go, connecting the legal map to the
substitute sites of lead 3.

## How the leads complement each other

| Lead | What it shifts | Which group responds (lead 2) | What it identifies |
|---|---|---|---|
| 1 | Gold price received (laundering cost) | Both, strongest near permits | Supply response to price; first instrument for upstream mining |
| 3 | Relative enforcement cost at a site | Outside crews | Mobility and displacement elasticity; second instrument |
| 4 | Legal status and pending protection | Crews (where they go); residents (participation deals) | Location choice; third instrument |
| 2 | The model linking them | Both | Which group dominates; downstream spread; relapse |

1. **Three independent shocks for one health effect.**
   - The current Amazon result rests on 36 onsets of unknown origin.
   - Leads 1, 3 and 4 each move upstream mining for a different reason, with different
     confounds.
   - A similar health effect per unit of upstream mining across all three is much
     stronger evidence than any single design.
   - If the effects differ, lead 2 suggests why: mechanized crew mining uses more mercury
     than local mining.
2. **Lead 2 turns the shocks into a test of mechanism.** Crews respond to laundering and
   enforcement and are displaced. Residents respond to lost fishing and drought and
   relapse. Only contamination produces the downstream asymmetry. Finding the predicted
   pattern across shocks is what makes this an economics paper rather than three reduced
   forms.
3. **Leads 3 and 4 complete each other's geography.** Lead 3 shows crews leaving
   targeted areas; lead 4 shows where weak protection lets them land. Together with the
   river network, they map who downstream bears the health cost.
4. **One shared data layer serves all four.** The new layer links to the existing
   river-stretch and municipality IDs:
   - the MapBiomas mining module;
   - mining permits and royalty payments;
   - IBAMA enforcement records;
   - Indigenous-land and protected-area boundaries;
   - malaria surveillance records;
   - fisher counts.
5. **One policy question.** Which lever reduces downstream harm per unit of effort:
   controlling gold flows (lead 1), destroying equipment on site (lead 3), or completing
   demarcation (lead 4)? Lead 2 explains why on-site enforcement alone relapses where
   livelihoods have collapsed.

## Proposed packaging and next steps

**Core paper (combining the pollution and health evidence with an economic driver).**
Lead 1 is the main design, lead 2 provides the model and mechanism tests, and lead 3 is
a section on displacement and on where the health burden goes.

**Lead 4.** Either a section on where displaced mining lands, or a separate paper if the
demarcation timing holds up.

**Order of work.**

1. **Build the shared data layer.** Start with the MapBiomas mining module (garimpo vs.
   industrial, gold vs. other ores), then mining permits and royalties. Replace the
   project's single mining class with garimpo-gold mining and re-run the health mining
   arm as a first check.
2. **Request malaria microdata access early.** It decides how far lead 2 can go.
3. **Estimate lead 1.** Run the reduced form on mining onsets, then on downstream mercury
   violations and infant deaths.
4. **Add lead 3.** Shocks, substitute sites, and the displacement ratio.
5. **Add lead 4.** Demarcation stages and anticipation shocks.

## Data inventory

| Data | Source | Use | Status |
|---|---|---|---|
| River network, reachability and distance matrices | project (BHO/ANA) | downstream links, asymmetry test | in pipeline |
| Water quality incl. mercury, lead, cadmium violations | project (monitoring-station panel) | contamination outcome | in pipeline |
| Hospitalizations, births, deaths by municipality | DATASUS (SIH, SINASC, SIM) | health outcomes | in pipeline |
| Land cover incl. a single mining class | MapBiomas | current mining exposure | in pipeline |
| Mining module (garimpo vs. industrial, by ore) | MapBiomas | mining exposure by type | to add |
| Mining permits with type, ore, dates | National Mining Agency (SIGMINE) | laundering exposure (lead 1), anticipation (lead 4) | to add |
| Royalty payments (CFEM) | National Mining Agency | declared output, laundering measure | to add |
| Gold geology | Geological Survey of Brazil | deposit quality, controls | to add |
| Fines, embargoes, seizures | IBAMA | enforcement shocks (lead 3) | to add |
| Indigenous lands with stage and dates | FUNAI | lead 4 | to add |
| Protected areas with creation dates | National register of conservation units | lead 4 | to add |
| Malaria cases with occupation, infection site, residence | SIVEP-Malária | separating crews and residents (lead 2) | access to check |
| Registered fishers, closed-season payments | Fisheries registry, federal transparency portal | fishing dependence (lead 2) | to add |
| Gold price | world market series | price shock | to add |

## Other leads considered

Ranked below leads 1–4, mainly because they are crowded or do not use the project's
river and health data:

5. **Bolsonaro and Lula election shocks.** Best used as shocks within leads 1–3 rather
   than on their own: the variation is over time only, and the topic is heavily studied
   for deforestation.
6. **Rural credit rule (Resolution 3,545) and substitution into mining.** An open question
   with a known design, but a likely precise null given that gold prices dominate. Useful
   as a test of the residents' side within lead 2.
7. **Random federal corruption audits.** A clean randomization, but an unclear link to
   garimpo, since mining permits are federal. A quick side test.
8. **The CAR registry, land titling and pasture as a possession claim.** Important and
   data-rich, but crowded, and the health arm found no pasture effect to carry a health
   story.
9. **Fragmenting clearings below satellite detection thresholds.** Already documented by
   a recent study of the priority-municipality blacklist.
10. **Supply-chain rules pushing activity into mining.** Weak priors (cattle returns are
    far below soy) and no data yet. The 2026 soy-moratorium collapse becomes testable
    once MapBiomas covers 2026–27.

## Caveats

- **Unverified sources.** Several sources behind the original brainstorm are from
  2025–26 and still need checking before a paper relies on them: the blacklisting
  fragmentation study, the 2026 reviews, and the Greenpeace and Grist permit figures.
- **Policy dates.** The dates of the 2013 law, the 2023 Supreme Court decision and the
  electronic-invoice requirement need confirming.
- **Dredges.** River-dredge mining is largely invisible to land-cover data, so any design
  based on MapBiomas undercounts it.
