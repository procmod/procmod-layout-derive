use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, LitInt, Meta};

/// Derives a `read` method that maps a remote process's memory into a Rust struct.
///
/// Each field must have an `#[offset(N)]` attribute specifying its byte offset
/// from the base address. Fields may optionally have a `#[pointer_chain(a, b, ...)]`
/// attribute to follow a chain of pointers before reading the final value.
///
/// The generated method signature is:
///
/// ```ignore
/// pub fn read(process: &procmod_layout::Process, base: usize) -> procmod_layout::Result<Self>
/// ```
///
/// # Attributes
///
/// - `#[offset(N)]` - byte offset from base address (required on every field)
/// - `#[pointer_chain(a, b, ...)]` - intermediate pointer offsets to follow before reading
///
/// # Safety requirement
///
/// All field types must be valid for any bit pattern. Numeric primitives (`u8`,
/// `u32`, `f32`, etc.), fixed-size arrays of numeric types, and `#[repr(C)]`
/// structs composed of such types are safe. Types with validity invariants
/// (`bool`, `char`, enums, references) must not be used - read them as their
/// underlying integer type instead (e.g., `u8` for booleans).
///
/// # Example
///
/// ```ignore
/// use procmod_layout::GameStruct;
///
/// #[derive(GameStruct)]
/// struct Player {
///     #[offset(0x100)]
///     health: f32,
///     #[offset(0x200)]
///     #[pointer_chain(0x10, 0x8)]
///     damage_mult: f32,
/// }
/// ```
#[proc_macro_derive(GameStruct, attributes(offset, pointer_chain))]
pub fn derive_game_struct(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match impl_game_struct(&input) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn impl_game_struct(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let name = &input.ident;

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(f) => &f.named,
            _ => {
                return Err(syn::Error::new(
                    Span::call_site(),
                    "GameStruct only supports structs with named fields",
                ))
            }
        },
        _ => {
            return Err(syn::Error::new(
                Span::call_site(),
                "GameStruct can only be derived for structs",
            ))
        }
    };

    let mut field_reads = Vec::new();

    for field in fields {
        let field_name = field.ident.as_ref().unwrap();
        let field_ty = &field.ty;

        let offset = parse_offset_attr(field)?;
        let chain = parse_pointer_chain_attr(field)?;

        let read_expr = if let Some(offsets) = chain {
            gen_pointer_chain_read(field_ty, offset, &offsets)
        } else {
            gen_direct_read(field_ty, offset)
        };

        field_reads.push(quote! {
            #field_name: #read_expr
        });
    }

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    Ok(quote! {
        impl #impl_generics #name #ty_generics #where_clause {
            /// Reads this struct from a remote process's memory at the given base address.
            pub fn read(
                __procmod_process: &::procmod_layout::Process,
                __procmod_base: usize,
            ) -> ::procmod_layout::Result<Self> {
                Ok(Self {
                    #(#field_reads),*
                })
            }
        }
    })
}

fn parse_offset_attr(field: &syn::Field) -> syn::Result<u64> {
    for attr in &field.attrs {
        if attr.path().is_ident("offset") {
            let lit: LitInt = attr.parse_args()?;
            return lit.base10_parse();
        }
    }

    Err(syn::Error::new_spanned(
        field.ident.as_ref().unwrap(),
        "missing #[offset(N)] attribute",
    ))
}

fn parse_pointer_chain_attr(field: &syn::Field) -> syn::Result<Option<Vec<u64>>> {
    for attr in &field.attrs {
        if attr.path().is_ident("pointer_chain") {
            match &attr.meta {
                Meta::List(list) => {
                    let parsed: syn::punctuated::Punctuated<LitInt, syn::Token![,]> =
                        list.parse_args_with(syn::punctuated::Punctuated::parse_terminated)?;

                    if parsed.is_empty() {
                        return Err(syn::Error::new_spanned(
                            list,
                            "pointer_chain requires at least one offset",
                        ));
                    }

                    let offsets = parsed
                        .iter()
                        .map(|lit| lit.base10_parse())
                        .collect::<syn::Result<Vec<u64>>>()?;

                    return Ok(Some(offsets));
                }
                _ => {
                    return Err(syn::Error::new_spanned(
                        attr,
                        "expected #[pointer_chain(offset, ...)]",
                    ))
                }
            }
        }
    }
    Ok(None)
}

fn gen_direct_read(ty: &syn::Type, offset: u64) -> proc_macro2::TokenStream {
    let offset_lit = LitInt::new(&format!("{offset}"), Span::call_site());
    quote! {
        unsafe { __procmod_process.read::<#ty>(__procmod_base + #offset_lit)? }
    }
}

fn gen_pointer_chain_read(
    ty: &syn::Type,
    base_offset: u64,
    chain: &[u64],
) -> proc_macro2::TokenStream {
    let base_offset_lit = LitInt::new(&format!("{base_offset}"), Span::call_site());

    let mut steps = Vec::new();

    // read initial pointer
    let first_var = syn::Ident::new("__ptr_0", Span::call_site());
    steps.push(quote! {
        let #first_var: usize = unsafe {
            __procmod_process.read::<usize>(__procmod_base + #base_offset_lit)?
        };
    });

    // follow intermediate pointers (all except last offset)
    let last_idx = chain.len() - 1;
    for (i, &offset) in chain[..last_idx].iter().enumerate() {
        let prev_var = syn::Ident::new(&format!("__ptr_{i}"), Span::call_site());
        let next_var = syn::Ident::new(&format!("__ptr_{}", i + 1), Span::call_site());
        let offset_lit = LitInt::new(&format!("{offset}"), Span::call_site());
        steps.push(quote! {
            let #next_var: usize = unsafe {
                __procmod_process.read::<usize>(#prev_var + #offset_lit)?
            };
        });
    }

    // read final value at last chain offset
    let final_var = syn::Ident::new(&format!("__ptr_{last_idx}"), Span::call_site());
    let final_offset_lit = LitInt::new(&format!("{}", chain[last_idx]), Span::call_site());

    quote! {
        {
            #(#steps)*
            unsafe { __procmod_process.read::<#ty>(#final_var + #final_offset_lit)? }
        }
    }
}
