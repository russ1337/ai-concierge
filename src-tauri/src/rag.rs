use std::path::Path;

#[derive(Debug, serde::Deserialize)]
pub struct Event {
    pub title: String,
    pub date: String,
    pub description: String,
}

pub fn load_events(events_path: &Path) -> Result<Vec<Event>, String> {
    let bytes =
        std::fs::read(events_path).map_err(|e| format!("Failed to read events file: {}", e))?;
    let events: Vec<Event> =
        serde_json::from_slice(&bytes).map_err(|e| format!("Invalid events JSON: {}", e))?;
    Ok(events)
}

fn event_searchable_text(event: &Event) -> String {
    format!("{} {}", event.title, event.description).to_lowercase()
}

pub fn search_events<'a>(events: &'a [Event], query: &str, limit: usize) -> Vec<&'a Event> {
    let query_lower = query.to_lowercase();
    let query_words: Vec<&str> = query_lower
        .split_whitespace()
        .filter(|s| s.len() > 1)
        .collect();
    if query_words.is_empty() {
        return events.iter().take(limit).collect();
    }
    let mut scored: Vec<(usize, &Event)> = events
        .iter()
        .map(|e| {
            let text = event_searchable_text(e);
            let matches = query_words.iter().filter(|w| text.contains(*w)).count();
            (matches, e)
        })
        .filter(|(n, _)| *n > 0)
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0));
    scored.into_iter().take(limit).map(|(_, e)| e).collect()
}

pub fn format_events_for_prompt(events: &[&Event]) -> String {
    if events.is_empty() {
        return String::from("(No relevant events found.)");
    }
    events
        .iter()
        .map(|e| format!("- {} ({}) {}", e.title, e.date, e.description))
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn retrieve_context(events_path: &Path, query: &str, limit: usize) -> Result<String, String> {
    let events = load_events(events_path)?;
    let relevant = search_events(&events, query, limit);
    Ok(format_events_for_prompt(&relevant))
}
