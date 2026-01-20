import { createClient } from '@supabase/supabase-js';

// REPLACE THESE WITH YOUR KEYS FROM SUPABASE DASHBOARD
const supabaseUrl = 'https://bhlfsvqtiaxgjfiuwehu.supabase.co';
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJobGZzdnF0aWF4Z2pmaXV3ZWh1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njg5MzEzNTYsImV4cCI6MjA4NDUwNzM1Nn0.RonioCmCDpFLt0XvM8LvLlyCG9m2gCfWioRI6hBD5FU';

export const supabase = createClient(supabaseUrl, supabaseKey);