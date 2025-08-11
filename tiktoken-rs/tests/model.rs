use tiktoken_rs::model::get_context_size;

#[test]
fn test_finetuned_context_size() {
    assert_eq!(
        get_context_size("ft:gpt-3.5-turbo-0125:custom"),
        get_context_size("gpt-3.5-turbo-0125")
    );
    assert_eq!(
        get_context_size("ft:gpt-4o:custom"),
        get_context_size("gpt-4o")
    );
    assert_eq!(
        get_context_size("ft:gpt-5:custom"),
        get_context_size("gpt-5")
    );
    assert_eq!(
        get_context_size("ft:gpt-4.1:custom"),
        get_context_size("gpt-4.1")
    );
}

#[test]
fn test_o_series_context_size() {
    assert_eq!(get_context_size("o3-small"), 200_000);
    assert_eq!(get_context_size("o4"), 200_000);
}

#[test]
fn test_4_1_series_context_size() {
    assert_eq!(get_context_size("gpt-4.1"), 1_047_576);
    assert_eq!(get_context_size("gpt-4.1-mini"), 1_047_576);
}

#[test]
fn test_5_series_context_size() {
    assert_eq!(get_context_size("gpt-5"), 400_000);
    assert_eq!(get_context_size("gpt-5-nano"), 400_000);
}
