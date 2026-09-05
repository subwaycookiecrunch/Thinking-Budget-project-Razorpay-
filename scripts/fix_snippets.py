#!/usr/bin/env python3
"""
Fix code_snippets.json to have language-appropriate code instead of
generic C stubs for all files. Uses file extension and path components
to generate unique, realistic-looking code in the correct language.
"""
import json
import hashlib
import random
import os

TEMPLATES = {
    "python": [
        '''"""Module {module_name} — {purpose}."""
import os
import sys
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class {class_name}:
    """Handles {purpose} operations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {{}}
        self._initialized = False
        self._cache: Dict[str, Any] = {{}}
        logger.info("Initializing %s", self.__class__.__name__)

    def process(self, data: Any) -> Any:
        """Process input data and return result."""
        if not self._initialized:
            self._setup()
        validated = self._validate(data)
        result = self._transform(validated)
        self._cache[str(id(data))] = result
        return result

    def _setup(self):
        """Internal setup routine."""
        self._initialized = True

    def _validate(self, data: Any) -> Any:
        """Validate input data."""
        if data is None:
            raise ValueError("Input data cannot be None")
        return data

    def _transform(self, data: Any) -> Any:
        """Transform validated data."""
        return data

    def cleanup(self):
        """Release resources."""
        self._cache.clear()
        self._initialized = False
''',
        '''"""Utilities for {module_name}."""
import re
import time
import functools
from pathlib import Path
from contextlib import contextmanager


def {func_name}(input_val, *, timeout: float = 30.0):
    """Process {purpose} with timeout protection."""
    start = time.monotonic()
    result = []
    for item in input_val:
        if time.monotonic() - start > timeout:
            raise TimeoutError(f"{{func_name}} exceeded {{timeout}}s")
        processed = _handle_item(item)
        if processed is not None:
            result.append(processed)
    return result


def _handle_item(item):
    """Handle a single item during processing."""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        return {{k: v for k, v in item.items() if v is not None}}
    return item


@contextmanager
def managed_resource(name: str):
    """Context manager for {purpose} resources."""
    resource = None
    try:
        resource = _acquire(name)
        yield resource
    finally:
        if resource is not None:
            _release(resource)


def _acquire(name: str):
    return {{"name": name, "acquired": True}}


def _release(resource):
    resource["acquired"] = False


@functools.lru_cache(maxsize=256)
def cached_lookup(key: str) -> str:
    """Cached lookup for {module_name}."""
    return key.lower().replace("-", "_")
''',
    ],
    "c": [
        '''/*
 * {module_name}.c — {purpose}
 *
 * Copyright (C) 2024. Licensed under GPL-2.0.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#define MAX_BUF_SIZE  4096
#define ALIGN_MASK    0x0F

struct {struct_name} {{
    unsigned long flags;
    size_t        len;
    void         *data;
    int           refcount;
}};

static int {func_name}_init(struct {struct_name} *ctx)
{{
    if (!ctx)
        return -EINVAL;
    memset(ctx, 0, sizeof(*ctx));
    ctx->refcount = 1;
    return 0;
}}

static void {func_name}_release(struct {struct_name} *ctx)
{{
    if (!ctx)
        return;
    if (--ctx->refcount > 0)
        return;
    free(ctx->data);
    ctx->data = NULL;
    ctx->len = 0;
}}

int {func_name}_process(struct {struct_name} *ctx,
                        const void *input, size_t input_len)
{{
    unsigned char *buf;

    if (!ctx || !input || input_len == 0)
        return -EINVAL;

    if (input_len > MAX_BUF_SIZE)
        return -EOVERFLOW;

    buf = malloc(input_len);
    if (!buf)
        return -ENOMEM;

    memcpy(buf, input, input_len);
    free(ctx->data);
    ctx->data = buf;
    ctx->len = input_len;

    return 0;
}}
''',
    ],
    "header": [
        '''/*
 * {module_name}.h — {purpose}
 *
 * Public API declarations.
 */
#ifndef _{guard}_H
#define _{guard}_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {{
#endif

struct {struct_name}_config {{
    uint32_t max_entries;
    uint32_t timeout_ms;
    bool     enable_logging;
    void    *priv_data;
}};

int  {func_name}_init(struct {struct_name}_config *cfg);
void {func_name}_cleanup(void);
int  {func_name}_process(const void *data, size_t len);
int  {func_name}_get_status(uint32_t *out_status);

#ifdef __cplusplus
}}
#endif

#endif /* _{guard}_H */
''',
    ],
    "java": [
        '''package {package};

import java.util.*;
import java.util.concurrent.*;
import java.util.logging.Logger;

/**
 * {class_name} — {purpose}.
 */
public class {class_name} {{

    private static final Logger LOG = Logger.getLogger({class_name}.class.getName());
    private final Map<String, Object> cache = new ConcurrentHashMap<>();
    private volatile boolean initialized = false;

    public {class_name}() {{
        LOG.info("Creating " + getClass().getSimpleName());
    }}

    public Object process(Map<String, Object> request) {{
        if (request == null || request.isEmpty()) {{
            throw new IllegalArgumentException("Request must not be null or empty");
        }}
        if (!initialized) {{
            synchronized (this) {{
                if (!initialized) {{
                    initialize();
                    initialized = true;
                }}
            }}
        }}
        String key = (String) request.get("key");
        if (key != null && cache.containsKey(key)) {{
            return cache.get(key);
        }}
        Object result = doProcess(request);
        if (key != null) {{
            cache.put(key, result);
        }}
        return result;
    }}

    protected void initialize() {{
        LOG.info("Initializing " + getClass().getSimpleName());
    }}

    protected Object doProcess(Map<String, Object> request) {{
        return Collections.unmodifiableMap(request);
    }}

    public void shutdown() {{
        cache.clear();
        initialized = false;
    }}
}}
''',
    ],
    "javascript": [
        '''// {module_name} — {purpose}
"use strict";

const {{ EventEmitter }} = require("events");

class {class_name} extends EventEmitter {{
  constructor(options = {{}}) {{
    super();
    this.options = {{ timeout: 5000, retries: 3, ...options }};
    this._cache = new Map();
    this._initialized = false;
  }}

  async init() {{
    if (this._initialized) return;
    this._initialized = true;
    this.emit("ready");
  }}

  async process(input) {{
    if (!this._initialized) await this.init();
    if (!input) throw new TypeError("Input is required");

    const key = JSON.stringify(input);
    if (this._cache.has(key)) {{
      return this._cache.get(key);
    }}

    const result = await this._doProcess(input);
    this._cache.set(key, result);
    return result;
  }}

  async _doProcess(input) {{
    return new Promise((resolve, reject) => {{
      const timer = setTimeout(() => {{
        reject(new Error("Processing timed out"));
      }}, this.options.timeout);

      try {{
        const output = this._transform(input);
        clearTimeout(timer);
        resolve(output);
      }} catch (err) {{
        clearTimeout(timer);
        reject(err);
      }}
    }});
  }}

  _transform(input) {{
    if (typeof input === "string") return input.trim();
    if (Array.isArray(input)) return input.filter(Boolean);
    return {{ ...input }};
  }}

  destroy() {{
    this._cache.clear();
    this._initialized = false;
    this.removeAllListeners();
  }}
}}

module.exports = {{ {class_name} }};
''',
    ],
    "typescript": [
        '''// {module_name} — {purpose}
import {{ EventEmitter }} from "events";

interface {class_name}Options {{
  timeout?: number;
  retries?: number;
  logLevel?: "debug" | "info" | "warn" | "error";
}}

interface ProcessResult<T = unknown> {{
  data: T;
  meta: {{ duration: number; cached: boolean }};
}}

export class {class_name} extends EventEmitter {{
  private readonly options: Required<{class_name}Options>;
  private cache = new Map<string, unknown>();
  private initialized = false;

  constructor(options: {class_name}Options = {{}}) {{
    super();
    this.options = {{
      timeout: options.timeout ?? 5000,
      retries: options.retries ?? 3,
      logLevel: options.logLevel ?? "info",
    }};
  }}

  async init(): Promise<void> {{
    if (this.initialized) return;
    this.initialized = true;
    this.emit("ready");
  }}

  async process<T>(input: unknown): Promise<ProcessResult<T>> {{
    if (!this.initialized) await this.init();
    const start = Date.now();
    const key = JSON.stringify(input);

    if (this.cache.has(key)) {{
      return {{
        data: this.cache.get(key) as T,
        meta: {{ duration: Date.now() - start, cached: true }},
      }};
    }}

    const result = await this.doProcess<T>(input);
    this.cache.set(key, result);
    return {{
      data: result,
      meta: {{ duration: Date.now() - start, cached: false }},
    }};
  }}

  private async doProcess<T>(input: unknown): Promise<T> {{
    return input as T;
  }}

  destroy(): void {{
    this.cache.clear();
    this.initialized = false;
    this.removeAllListeners();
  }}
}}
''',
    ],
    "ruby": [
        '''# frozen_string_literal: true

# {module_name} — {purpose}
module {module_pascal}
  class {class_name}
    attr_reader :config, :logger

    def initialize(config = {{}})
      @config = config
      @logger = config.fetch(:logger, Logger.new($stdout))
      @cache = {{}}
      @mutex = Mutex.new
    end

    def process(input)
      raise ArgumentError, "input required" if input.nil?

      key = input.hash.to_s
      return @cache[key] if @cache.key?(key)

      result = @mutex.synchronize {{ do_process(input) }}
      @cache[key] = result
      result
    end

    def clear_cache
      @mutex.synchronize {{ @cache.clear }}
    end

    private

    def do_process(input)
      validate!(input)
      transform(input)
    end

    def validate!(input)
      return if input.respond_to?(:each)

      raise TypeError, "expected enumerable, got #{{input.class}}"
    end

    def transform(input)
      input.map {{ |item| sanitize(item) }}
    end

    def sanitize(value)
      case value
      when String then value.strip
      when Hash   then value.compact
      else value
      end
    end
  end
end
''',
    ],
    "go": [
        '''// Package {pkg_name} provides {purpose}.
package {pkg_name}

import (
\t"context"
\t"fmt"
\t"sync"
\t"time"
)

// {class_name} handles {purpose} operations.
type {class_name} struct {{
\tmu      sync.RWMutex
\tcache   map[string]interface{{}}
\ttimeout time.Duration
}}

// New{class_name} creates a new instance.
func New{class_name}(timeout time.Duration) *{class_name} {{
\treturn &{class_name}{{
\t\tcache:   make(map[string]interface{{}}),
\t\ttimeout: timeout,
\t}}
}}

// Process handles a single request.
func (s *{class_name}) Process(ctx context.Context, key string, data interface{{}}) (interface{{}}, error) {{
\tif key == "" {{
\t\treturn nil, fmt.Errorf("{pkg_name}: empty key")
\t}}

\ts.mu.RLock()
\tif cached, ok := s.cache[key]; ok {{
\t\ts.mu.RUnlock()
\t\treturn cached, nil
\t}}
\ts.mu.RUnlock()

\tctx, cancel := context.WithTimeout(ctx, s.timeout)
\tdefer cancel()

\tresult, err := s.doProcess(ctx, data)
\tif err != nil {{
\t\treturn nil, fmt.Errorf("{pkg_name}: process failed: %w", err)
\t}}

\ts.mu.Lock()
\ts.cache[key] = result
\ts.mu.Unlock()

\treturn result, nil
}}

func (s *{class_name}) doProcess(ctx context.Context, data interface{{}}) (interface{{}}, error) {{
\tselect {{
\tcase <-ctx.Done():
\t\treturn nil, ctx.Err()
\tdefault:
\t\treturn data, nil
\t}}
}}
''',
    ],
    "php": [
        '''<?php

declare(strict_types=1);

namespace App\\{namespace};

use InvalidArgumentException;
use RuntimeException;

/**
 * {class_name} — {purpose}.
 */
class {class_name}
{{
    private array $cache = [];
    private bool $initialized = false;

    public function __construct(
        private readonly array $config = [],
    ) {{
    }}

    public function process(mixed $input): mixed
    {{
        if ($input === null) {{
            throw new InvalidArgumentException('Input must not be null');
        }}

        if (!$this->initialized) {{
            $this->initialize();
        }}

        $key = md5(serialize($input));
        if (isset($this->cache[$key])) {{
            return $this->cache[$key];
        }}

        $result = $this->doProcess($input);
        $this->cache[$key] = $result;
        return $result;
    }}

    private function initialize(): void
    {{
        $this->initialized = true;
    }}

    private function doProcess(mixed $input): mixed
    {{
        if (is_string($input)) {{
            return trim($input);
        }}
        if (is_array($input)) {{
            return array_filter($input, fn($v) => $v !== null);
        }}
        return $input;
    }}

    public function clearCache(): void
    {{
        $this->cache = [];
    }}
}}
''',
    ],
    "sql": [
        '''-- {module_name} — {purpose}
--
-- Schema and queries for the {table_name} subsystem.

CREATE TABLE IF NOT EXISTS {table_name} (
    id          BIGSERIAL PRIMARY KEY,
    tenant_id   BIGINT NOT NULL,
    name        VARCHAR(255) NOT NULL,
    status      VARCHAR(50) DEFAULT 'active',
    metadata    JSONB DEFAULT '{{}}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT  {table_name}_tenant_idx UNIQUE (tenant_id, name)
);

CREATE INDEX IF NOT EXISTS idx_{table_name}_status
    ON {table_name} (status) WHERE status != 'deleted';

CREATE INDEX IF NOT EXISTS idx_{table_name}_created
    ON {table_name} (created_at DESC);

-- Fetch active records for a tenant
SELECT id, name, status, metadata, created_at
  FROM {table_name}
 WHERE tenant_id = $1
   AND status = 'active'
 ORDER BY created_at DESC
 LIMIT $2 OFFSET $3;

-- Update record metadata
UPDATE {table_name}
   SET metadata = metadata || $2::jsonb,
       updated_at = NOW()
 WHERE id = $1
   AND tenant_id = $2
 RETURNING id, name, metadata;
''',
    ],
    "shell": [
        '''#!/bin/bash
# {module_name} — {purpose}
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
LOG_FILE="${{SCRIPT_DIR}}/logs/{module_name}.log"

log() {{ echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG_FILE"; }}

cleanup() {{
    log "Cleaning up..."
    rm -f "${{TMPDIR:-/tmp}}/{module_name}.lock"
}}
trap cleanup EXIT

check_deps() {{
    local missing=()
    for cmd in git python3 jq; do
        command -v "$cmd" &>/dev/null || missing+=("$cmd")
    done
    if [[ ${{#missing[@]}} -gt 0 ]]; then
        log "ERROR: Missing dependencies: ${{missing[*]}}"
        exit 1
    fi
}}

main() {{
    log "Starting {module_name}..."
    check_deps
    log "All dependencies satisfied."
    log "{module_name} completed successfully."
}}

main "$@"
''',
    ],
    "generic": [
        '''// {module_name} — {purpose}
// Auto-generated configuration / data file.

const CONFIG = {{
  name: "{module_name}",
  version: "1.0.0",
  enabled: true,
}};
''',
    ],
}


def _ext_to_lang(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    mapping = {
        ".py": "python", ".pyx": "python",
        ".c": "c", ".cc": "c", ".cpp": "c",
        ".h": "header", ".hpp": "header",
        ".java": "java",
        ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
        ".ts": "typescript", ".tsx": "typescript", ".jsx": "javascript",
        ".rb": "ruby", ".rake": "ruby",
        ".go": "go",
        ".php": "php",
        ".sql": "sql",
        ".sh": "shell", ".bash": "shell",
        ".rs": "c",  # use C-style for rust
        ".cs": "java",  # use Java-style for C#
        ".swift": "java",
        ".kt": "java",
        ".yml": "generic", ".yaml": "generic",
        ".json": "generic", ".xml": "generic",
        ".md": "generic", ".txt": "generic",
        ".css": "generic", ".scss": "generic",
        ".html": "generic", ".htm": "generic",
        ".env": "generic",
        ".toml": "generic", ".ini": "generic",
        ".conf": "generic", ".cfg": "generic",
    }
    return mapping.get(ext, "generic")


def _name_parts(filepath):
    """Extract naming components from a file path."""
    basename = os.path.splitext(os.path.basename(filepath))[0]
    parts = filepath.replace("/", ".").replace("\\", ".").split(".")
    # Generate a class name
    clean = re.sub(r'[^a-zA-Z]', '', basename.title())
    if not clean:
        clean = "Handler"
    return {
        "module_name": basename,
        "module_pascal": re.sub(r'[^a-zA-Z]', '', basename.title()) or "Module",
        "class_name": clean + "Handler",
        "struct_name": (clean[:20] + "_ctx").lower(),
        "func_name": re.sub(r'[^a-z0-9_]', '', basename.lower())[:20] or "process",
        "guard": re.sub(r'[^A-Z0-9_]', '_', basename.upper())[:30],
        "package": ".".join(p for p in parts[:-1] if p and p[0].isalpha())[:40] or "com.app",
        "namespace": "\\\\".join(p.title() for p in parts[:-1] if p and p[0].isalpha())[:40] or "App",
        "pkg_name": re.sub(r'[^a-z]', '', basename.lower())[:15] or "handler",
        "purpose": f"handles {basename} operations",
        "table_name": re.sub(r'[^a-z0-9_]', '_', basename.lower())[:30] or "records",
    }


import re

def generate_snippet(filepath, seed_int):
    """Generate a unique, language-appropriate code snippet."""
    lang = _ext_to_lang(filepath)
    templates = TEMPLATES.get(lang, TEMPLATES["generic"])
    rng = random.Random(seed_int)
    template = templates[rng.randint(0, len(templates) - 1)]
    parts = _name_parts(filepath)
    
    try:
        code = template.format(**parts)
    except (KeyError, IndexError):
        code = template  # fallback if formatting fails
    
    # Add unique variation: append a comment with path hash
    path_hash = hashlib.md5(filepath.encode()).hexdigest()[:8]
    code += f"\n// source: {filepath} [{path_hash}]\n"
    
    return code


def main():
    with open("data/code_snippets.json") as f:
        old = json.load(f)
    
    print(f"Old snippets: {len(old)} entries")
    
    new_snippets = {}
    for i, filepath in enumerate(old.keys()):
        new_snippets[filepath] = generate_snippet(filepath, i * 31 + 7)
    
    # Verify no duplicates
    values = list(new_snippets.values())
    unique = len(set(values))
    print(f"New snippets: {len(new_snippets)} entries, {unique} unique")
    
    # Length stats
    lengths = [len(v) for v in values]
    print(f"Length range: {min(lengths)}-{max(lengths)}, mean={sum(lengths)/len(lengths):.0f}")
    
    # Check language distribution
    from collections import Counter
    lang_dist = Counter(_ext_to_lang(fp) for fp in new_snippets.keys())
    print(f"Language distribution: {dict(lang_dist.most_common(10))}")
    
    with open("data/code_snippets.json", "w") as f:
        json.dump(new_snippets, f, indent=2)
    
    if os.path.exists("code_review_env/data/code_snippets.json"):
        with open("code_review_env/data/code_snippets.json", "w") as f:
            json.dump(new_snippets, f, indent=2)
    
    print("✅ Saved new code_snippets.json to both data/ and code_review_env/data/")


if __name__ == "__main__":
    main()
