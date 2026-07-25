-- Run this once in Supabase: SQL Editor -> New query -> Run.
create table if not exists books (
  book_id text primary key,
  display_name text not null,
  created_at timestamptz default now()
);

create table if not exists threads (
  thread_id uuid primary key default gen_random_uuid(),
  book_id text not null references books(book_id) on delete cascade,
  title text not null default 'New chat',
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table if not exists messages (
  message_id uuid primary key default gen_random_uuid(),
  thread_id uuid not null references threads(thread_id) on delete cascade,
  role text not null check (role in ('user', 'assistant')),
  content text not null,
  routed_sections jsonb,
  images_used jsonb,
  created_at timestamptz default now()
);

create index if not exists threads_book_id_idx on threads (book_id);
create index if not exists messages_thread_created_at_idx on messages (thread_id, created_at);

-- The app uses the service-role key only on the Streamlit server.  Keeping
-- RLS on prevents somebody with the browser-visible anonymous key from
-- reading or changing chat history directly through Supabase's REST API.
alter table books enable row level security;
alter table threads enable row level security;
alter table messages enable row level security;
