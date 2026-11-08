# Running Grafana against the store

```
make grafana        # then open http://localhost:3000
```

Boards appear under the **Offchart** folder.

That runs `grafana/docker-compose.yml`, which is stateless by design — no volume for
Grafana's own database, because everything defining the boards is committed in this repo
and reprovisioned on every start. Delete the container and nothing is lost.

The container installs the SQLite plugin itself via `GF_PLUGINS_PREINSTALL`, so the
first start takes about 20 seconds. Anonymous access is on and there is no login prompt;
that is fine for one reader on a laptop and should be turned off before the port is
exposed anywhere.

```
make grafana-stop    stop it
make grafana-logs    follow its logs
```

## Installing the plugin into a Grafana you already run

If you would rather not use the container, three steps:

```
grafana-cli plugins install frser-sqlite-datasource    # or Administration > Plugins
systemctl restart grafana-server                       # brew services restart grafana, etc.
```

Then point Grafana at this repo's committed provisioning. Either copy the two
directories into your Grafana provisioning path:

```
cp grafana/provisioning/datasources/sqlite.yaml   /etc/grafana/provisioning/datasources/
cp grafana/provisioning/dashboards/offchart.yaml  /etc/grafana/provisioning/dashboards/
```

…or add the datasource by hand: type **SQLite**, path = the absolute path to
`data/offchart.sqlite`, and import the files in `grafana/dashboards/`.

**Two edits are required if you go native**, because the committed files use container
paths: `path:` in `sqlite.yaml` and `options.path` in `offchart.yaml` both say `/repo/…`
and must become real paths on your machine.

## Notes on the plugin, since it is the one third-party piece here

- `frser-sqlite-datasource` is a **signed community plugin**, one maintainer, catalog
  badge "Needs attention". It is pure Go via `modernc.org/sqlite`, so it needs no cgo
  and runs on the default Alpine Grafana image without
  `allow_loading_unsigned_plugins`.
- It reads a time column as **epoch seconds**, so every panel query divides:
  `ts/1000 AS time`. See the symptom below.
- It implements one macro, `$__unixEpochGroupSeconds`. There is no `$__timeFilter`, so
  panels write their own bound: `ts BETWEEN $__from AND $__to`, with `ts` stored in
  milliseconds because that is what Grafana's `$__from` / `$__to` are.
- Template variables use the plugin's own `__text` / `__value` convention (its own, not
  Grafana's, and only since plugin v3.5.0). The dropdown **fuzzy-matches the label
  only**, which is why every market label carries both the name and the code.
- If it ever stops being viable, the schema is ordinary SQL and moving these tables into
  Postgres is an afternoon.

## Two symptoms worth recognising

Both are failures that look like something else, so they are written down here rather
than only in the code that prevents them.

**A chart starting in 1953.** Passing the stored millisecond `ts` straight to the plugin
instead of `ts/1000` renders 1953-11-19 where the data says 2026-11-03 — a wrong chart
that looks like a real one. `tests/test_grafana.py` asserts the division on every panel.

**One blank panel beside two full ones.** This is journal mode, not missing data. WAL is
right for writing — one writer, N readers — but with the file bind-mounted into the
Grafana container on macOS, three panels querying concurrently made one lose a lock
race: `database is locked (5) (SQLITE_BUSY)`. So every writer in this repo hands the file
back in rollback-journal mode when it finishes; see `lib/db.finalize_for_serving`. If you
see this, something crashed mid-write — re-run `make pull` and it will be handed back
correctly.

## The dashboards are the committed JSON

Provisioned with `allowUiUpdates: false`, so Grafana refuses UI saves. Edit the JSON in
`grafana/dashboards/` and restart, or export from the UI and commit the result.

Moving to Grafana cost this repo the ability to assert on a rendered figure, so what is
left is that the JSON is authoritative and `tests/test_grafana.py` reads it: no panel
may rank in SQL, none may read an archive table, every series must declare its axis
side, and the time column must be divided.

That axis lint is written as a **positive** assertion, and the reason is a real trap:
Grafana omits default-valued keys, and `axisPlacement` defaults to `auto`, which means
"first field left, everything else right". A lint looking for `"right"` passes on a
dashboard that visibly renders a second y-axis.
