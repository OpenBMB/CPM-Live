from collections import OrderedDict
import json
import multiprocessing
import os
from queue import Empty
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict, Union
from cpm_live.dataset import DistributedDataset
from cpm_live.tokenizers import CPMAntTokenizer
import numpy as np
import time
from numpy._typing import NDArray
import torch
import bmtrain as bmt

class _MixedDatasetConfig(TypedDict):
    weight : float
    path : str
    transforms : List[Dict[str, Any]]
    task_name : str
    dataset_name : str

    lines : int
    dataset : DistributedDataset

class _DictTree(TypedDict):
    value : str
    children : List['_DictTree']
    depth : int
    segment_id : int
    need_predict : bool

class CPMBeeBatch(TypedDict):
    inputs : NDArray[np.int32]
    length : NDArray[np.int32]
    context : NDArray[np.bool_]
    sample_ids : NDArray[np.int32]
    num_segments : NDArray[np.int32]
    segment_ids : NDArray[np.int32]
    segment_rel_offset : NDArray[np.int32]
    segment_rel : NDArray[np.int32]
    spans : NDArray[np.int32]
    target : NDArray[np.int32]
    task_ids : NDArray[np.int32]
    task_names : List[str]

class _MixedDatasetBatchPacker:
    def __init__(self,
            batch_size : int,
            max_length : int,
            tokenizer : CPMAntTokenizer,
            incontext_sample_weight : Union[List[float], np.ndarray],
            max_depth : int = 16,
        ) -> None:
        self._batch_size = batch_size
        self._max_length = max_length
        self._sample_weight = np.array(incontext_sample_weight, dtype=np.float32)
        self._max_depth = max_depth
        self.tokenizer = tokenizer

        self._inputs : List[NDArray[np.int32]] = []
        self._context : List[NDArray[np.int8]] = []
        self._sample_ids : List[NDArray[np.int32]] = []
        self._segments : List[NDArray[np.int32]] = []
        self._num_segments : List[NDArray[np.int32]] = []
        self._segment_rel_offset : List[NDArray[np.int32]] = []
        self._segment_rel : List[NDArray[np.int32]] = []
        self._spans : List[List[int]] = []
        self._task_ids  : List[List[str]] = []
    
    def apply_transform(
            self,
            data : Dict[str, Any],
            transform : Optional[Dict[str, Any]]
        ):
        if transform is None:
            return data

        mapping_list : List[Tuple[str, str]] = []
        def _walk_transform_dict(data : Union[Dict[str, Any], str], prefix : str = ""):
            if isinstance(data, dict):
                for k, v in data.items():
                    if len(prefix) > 0:
                        _walk_transform_dict(v, prefix + "." + k)
                    else:
                        _walk_transform_dict(v, k)
            else:
                assert isinstance(data, str), "Invalid transform {}".format(data)
                mapping_list.append((prefix, data))
        _walk_transform_dict(transform)

        expanded_mapping_list : List[Tuple[str, Any]] = []
        def _expand_mapping(data : Any, stars : List[str], path : List[str], target : List[str]):
            if len(path) == 0:
                num_stars = 0
                for it in target:
                    if it == "*":
                        num_stars += 1
                if num_stars != len(stars):
                    raise ValueError("Invalid transform {}".format(".".join(target)))
                
                nw_tgt = []
                num_stars = 0
                for it in target:
                    if it == "*":
                        nw_tgt.append(stars[num_stars])
                        num_stars += 1
                    else:
                        nw_tgt.append(it)
                expanded_mapping_list.append(
                    (".".join(nw_tgt), data)
                )
            else:
                if not isinstance(data, dict):
                    raise ValueError("Invalid data {}".format(data))
                if path[0] == "*":
                    for k, v in data.items():
                        _expand_mapping(v, stars + [k], path[1:], target)
                else:
                    _expand_mapping(data[path[0]], stars, path[1:], target)
        
        # expand mapping list
        for tgt, src in mapping_list:
            if src.startswith("$"):
                # copy from src
                _expand_mapping(data, [], src[1:].split("."), tgt.split("."))
            else:
                if "*" in tgt:
                    raise ValueError("Constant value is not allowed to have `*` in prefix")
                expanded_mapping_list.append((tgt, src))
        
        ret = {}
        for tgt, val in expanded_mapping_list:
            tgt = tgt.split(".")
            cur = ret
            while len(tgt) > 1:
                cur = cur[tgt[0]]
                tgt = tgt[1:]
            cur[tgt[0]] = val
        return ret
    
    def rel_to_bucket(self, n_up, n_down):
        ret = n_up * self._max_depth + n_down
        if ret == 0:
            return ret
        else:
            # bucket 1 is reserved for incontext samples
            return ret + 1
    
    def data_to_id(self, data : Any):
        root : _DictTree = {
            "value" : "<root>",
            "children" : [],
            "depth" : 0,
            "segment_id" : 0,
            "need_predict": False
        }

        segments = [root]
        def _build_dict_tree(data : Any, depth : int, need_predict : bool) -> List[_DictTree]:
            if isinstance(data, dict):
                ret : List[_DictTree] = []
                for k, v in data.items():
                    child_info = {
                        "value" : k,
                        "children" : [],
                        "depth" : depth,
                        "segment_id" : len(segments),
                        "need_predict" : False    # only leaves are contexts
                    }
                    segments.append(child_info)
                    child_info["children"] = _build_dict_tree(v, depth + 1, need_predict or (depth == 1 and k == "answer")) # elements in <root>.answer

                    ret.append(child_info)
                return ret
            else:
                assert isinstance(data, str), "Invalid data {}".format(data)
                ret = _DictTree = {
                    "value": data,
                    "children" : [],
                    "depth" : depth,
                    "segment_id" : len(segments),
                    "need_predict": need_predict
                }
                segments.append(ret)
                return [ret]
        root["children"] = _build_dict_tree(data, 1, False)
        
        num_segments = len(segments)
        segment_rel = np.zeros((num_segments * num_segments,), dtype=np.int32)
        def _build_segment_rel(node : _DictTree) -> List[Tuple[int, int]]:
            ret : List[Tuple[int, int]] = [(node["segment_id"], node["depth"])]
            for child in node["children"]:
                sub = _build_segment_rel(child)
                for seg_id_1, depth_1 in sub:
                    for seg_id_2, depth_2 in ret:
                        n_up = min(node["depth"] - depth_1, self._max_depth - 1)
                        n_down = min(node["depth"] - depth_2, self._max_depth - 1)
                        segment_rel[seg_id_1 * num_segments + seg_id_2] = self.rel_to_bucket(n_up, n_down)
                        segment_rel[seg_id_2 * num_segments + seg_id_1] = self.rel_to_bucket(n_down, n_up)
                ret.extend(sub)
            return ret
        _build_segment_rel(root)

        input_ids : List[int] = []
        segment_bound : List[Tuple[int, int]] = []
        for seg in segments:
            tokens = [ self.tokenizer.bos_id ] + self.tokenizer.encode(seg["value"]) + [ self.tokenizer.eos_id ]
            begin = len(input_ids)
            input_ids.extend(tokens)
            end = len(input_ids)
            segment_bound.append((begin, end))
        
        ids = np.array(input_ids, dtype=np.int32)
        segs = np.zeros((ids.shape[0],), dtype=np.int32)
        context = np.zeros((ids.shape[0],), dtype=np.int8)
        for i, (begin, end) in enumerate(segment_bound):
            if segments[i]["need_predict"]:
                context[begin:end] = 0
            else:
                context[begin:end] = 1
            segs[begin:end] = i
        return ids, context, segs, segment_rel, num_segments
    
    def build_instance(self, config : _MixedDatasetConfig):
        num_incontext = np.random.choice(self._sample_weight.shape[0], p=self._sample_weight)
        ds = config["dataset"]
        transforms = config["transforms"]
        transform_id = np.random.choice(len(transforms) + 1)
        if transform_id == len(transforms):
            transform = None
        else:
            transform = transforms[transform_id]
        
        while True:
            inp = ds.read()
            inp = self.apply_transform(inp, transform)

            input_ids, context, segment_ids, segment_rel, n_segments = self.data_to_id(inp)
            input_ids = input_ids[:self._max_length]
            context = context[:self._max_length]
            segment_ids = segment_ids[:self._max_length]
            
            if (context == 0).any():
                # some values are not in context
                break

        sample_ids = np.zeros(input_ids.shape, dtype=np.int32)
        segment_rel_offset = np.zeros(input_ids.shape, dtype=np.int32)
        num_segments = np.full(input_ids.shape, n_segments, dtype=np.int32)

        for i in range(num_incontext):
            if input_ids.shape[0] >= self._max_length:
                # early break
                break

            sample = ds.read()
            sample = self.apply_transform(sample, transform)
            sample_input_ids, _, sample_segments, sample_rel, n_segments = self.data_to_id(sample)

            if input_ids.shape[0] + sample_input_ids.shape[0] > self._max_length:
                # too long, break
                break

            input_ids = np.concatenate([input_ids, sample_input_ids], axis=0)
            context = np.concatenate([context, np.ones(sample_input_ids.shape, dtype=np.int8)], axis=0)
            segment_ids = np.concatenate([segment_ids, sample_segments], axis=0)
            segment_rel_offset = np.concatenate([segment_rel_offset, np.full(sample_input_ids.shape, segment_rel.shape[0], dtype=np.int32)], axis=0)
            segment_rel = np.concatenate([segment_rel, sample_rel], axis=0)
            sample_ids = np.concatenate([sample_ids, np.full(sample_input_ids.shape, i + 1, dtype=np.int32)], axis=0)
            num_segments = np.concatenate([num_segments, np.full(sample_input_ids.shape, n_segments, dtype=np.int32)], axis=0)
        return input_ids, context, segment_ids, segment_rel_offset, segment_rel, sample_ids, num_segments
    
    def add_data(self, config : _MixedDatasetConfig) -> Optional[CPMBeeBatch]:
        input_ids, context, segment_ids, segment_rel_offset, segment_rel, sample_ids, num_segments = self.build_instance(config)

        # add to batch
        best_fit : Union[None, int] = None
        best_fit_space : Union[None, int] = None
        for i in range(len(self._inputs)):
            space = self._max_length - self._inputs[i].shape[0]
            if input_ids.shape[0] <= space:
                if best_fit is None:
                    best_fit = i
                    best_fit_space = space
                elif best_fit_space > space:
                    best_fit = i
                    best_fit_space = space
        if best_fit is None:
            # add a new instance
            self._inputs.append(input_ids)
            self._context.append(context)
            self._sample_ids.append(sample_ids)
            self._segments.append(segment_ids)
            self._num_segments.append(num_segments)
            self._segment_rel_offset.append(segment_rel_offset)
            self._segment_rel.append(segment_rel)
            self._spans.append([input_ids.shape[0]])
            self._task_ids.append([config["task_name"]])
        else:
            # add to existing instance
            self._inputs[best_fit] = np.concatenate([self._inputs[best_fit], input_ids], axis=0)
            self._context[best_fit] = np.concatenate([self._context[best_fit], context], axis=0)
            self._sample_ids[best_fit] = np.concatenate([self._sample_ids[best_fit], sample_ids], axis=0)
            self._segments[best_fit] = np.concatenate([self._segments[best_fit], segment_ids], axis=0)
            self._num_segments[best_fit] = np.concatenate([self._num_segments[best_fit], num_segments], axis=0)
            self._segment_rel_offset[best_fit] = np.concatenate([
                self._segment_rel_offset[best_fit], 
                segment_rel_offset + self._segment_rel[best_fit].shape[0]
            ], axis=0)
            self._segment_rel[best_fit] = np.concatenate([self._segment_rel[best_fit], segment_rel], axis=0)
            self._spans[best_fit].append(self._inputs[best_fit].shape[0])
            self._task_ids[best_fit].append(config["task_name"])

        if len(self._inputs) > self._batch_size:
            # pack batch
            inputs = np.zeros((self._batch_size, self._max_length), dtype=np.int32)
            context = np.zeros((self._batch_size, self._max_length), dtype=np.int8)
            sample_ids = np.zeros((self._batch_size, self._max_length), dtype=np.int32)
            segments = np.zeros((self._batch_size, self._max_length), dtype=np.int32)
            num_segments = np.zeros((self._batch_size, self._max_length), dtype=np.int32)
            segment_rel_offset = np.zeros((self._batch_size, self._max_length), dtype=np.int32)
            tgt = np.full((self._batch_size, self._max_length), -100, dtype=np.int32)

            max_rel = 0
            for i in range(self._batch_size):
                max_rel = max(max_rel, self._segment_rel[i].shape[0])
            segment_rel = np.zeros((self._batch_size, max_rel), dtype=np.int32)
            spans = np.zeros((self._batch_size, self._max_length), dtype=np.int32)
            length = np.zeros(self._batch_size, dtype=np.int32)
            task_ids = np.zeros(self._batch_size, self._max_length, dtype=np.int32)

            all_task_names : Set[str] = set()
            for i in range(self._batch_size):
                for task_name in self._task_ids[i]:
                    all_task_names.add(task_name)
            task_names : List[str] = list(all_task_names)
            task_name_to_id = {
                name : i for i, name in enumerate(task_names)
            }

            for i in range(self._batch_size):
                l = self._inputs[i].shape[0]
                rel_size = self._segment_rel[i].shape[0]
                inputs[i, :l] = self._inputs[i]
                context[i, :l] = self._context[i]
                sample_ids[i, :l] = self._sample_ids[i]
                segments[i, :l] = self._segments[i]
                num_segments[i, :l] = self._num_segments[i]
                segment_rel_offset[i, :l] = self._segment_rel_offset[i]
                segment_rel[i, :rel_size] = self._segment_rel[i]
                
                span_begin = 0
                for i, (span_end, task_name) in enumerate(zip(self._spans[i], self._task_ids[i])):
                    spans[i, span_begin:span_end] = i
                    task_ids[i, span_begin:span_end] = task_name_to_id[task_name]
                    span_begin = span_end
                length[i] = l
                tgt[i, 0 : l - 1] = np.where(
                    context[i, 1 : l] > 0,
                    -100,
                    inputs[i, 1 : l]
                )
            
            self._inputs = self._inputs[self._batch_size:]
            self._context = self._context[self._batch_size:]
            self._sample_ids = self._sample_ids[self._batch_size:]
            self._segments = self._segments[self._batch_size:]
            self._num_segments = self._num_segments[self._batch_size:]
            self._segment_rel_offset = self._segment_rel_offset[self._batch_size:]
            self._segment_rel = self._segment_rel[self._batch_size:]
            self._spans = self._spans[self._batch_size:]
            self._task_ids = self._task_ids[self._batch_size:]
            return {
                "inputs" : inputs,
                "length": length,
                "context" : context > 0,
                "sample_ids" : sample_ids,
                "num_segments" : num_segments,
                "segment_ids" : segments,
                "segment_rel_offset" : segment_rel_offset,
                "segment_rel" : segment_rel,
                "spans" : spans,
                "target": tgt,
                "task_ids": task_ids,
                "task_names": task_names
            }
        else:
            # not ready
            return None


class _MixedDatasetConfigMananger:
    def __init__(self, config_path : str) -> None:
        self._config_path = config_path
        self._config = None
        self._last_m = 0

    def changed(self):
        m_time = os.stat(self._config_path).st_mtime
        if m_time > self._last_m:
            # try to load new config
            try:
                self._config = json.load(open(self._config_path, "r", encoding="utf-8"))
            except Exception:
                # failed to load config
                return False
            
            # new config loaded
            self._last_m = m_time
            return True
        return False
    
    def get_config(self) -> List[_MixedDatasetConfig]:
        if self._config is None:
            if not self.changed():
                raise RuntimeError("Failed to load config")
        return self._config

def _mixed_dataset_process(
        config_path : str,
        q_cmd : multiprocessing.Queue,
        q_data : multiprocessing.Queue,
        rank : int,
        world_size : int,
        packer : _MixedDatasetBatchPacker,
    ):
    def _build_sample_weights(
            config : List[_MixedDatasetConfig]
        ):
        if len(config) == 0:
            return np.array([], dtype=np.float32)
        weights = [c["weight"] * c["lines"] for c in config]
        weights = np.array(weights, dtype=np.float32)
        sm_weight = weights.sum()
        if sm_weight > 0:
            weights = weights / sm_weight
            return weights
        else:
            raise RuntimeError("Empty datasets")

    cfg_mgr = _MixedDatasetConfigMananger(config_path)
    config = cfg_mgr.get_config()

    for c in config:
        ds = DistributedDataset(
            c["path"],
            rank,
            world_size,
        )
        
        c["lines"] = ds._nlines
        c["dataset"] = ds
        if "weight" not in c:
            c["weight"] = 1.0
        if "transforms" not in c:
            c["transforms"] = []
    
    weights = _build_sample_weights(config)

    should_stop = False
    should_start = False

    while not should_stop:
        # update config first
        if cfg_mgr.changed():
            path_ds_map : Dict[str, _MixedDatasetConfig] = {}
            nw_path_set : Set[str] = set()

            # load new config
            nw_config = cfg_mgr.get_config()

            # build path -> dataset map
            for c in config:
                path_ds_map[c["path"]] = c

            # add new datasets    
            for c in nw_config:
                if c["path"] in path_ds_map:
                    # update values only
                    if "weight" in c:
                        path_ds_map[c["path"]]["weight"] = c["weight"]
                    if "transform" in c:
                        path_ds_map[c["path"]]["transforms"] = c["transforms"]
                else:
                    # new dataset
                    ds = DistributedDataset(
                        c["path"],
                        rank,
                        world_size,
                    )
                    c["lines"] = ds._nlines
                    c["dataset"] = ds
                    if "weight" not in c:
                        c["weight"] = 1.0
                    path_ds_map[c["path"]] = c
                nw_path_set.add(c["path"])
            
            # remove unused datasets
            for c in config:
                if c["path"] not in nw_path_set:
                    del path_ds_map[c["path"]]
            
            config : List[_MixedDatasetConfig] = []
            for c in nw_config:
                config.append(path_ds_map[c["path"]])
            del path_ds_map
            del nw_path_set
            del nw_config
            

            weights = _build_sample_weights(config)

        # get cmds
        while True:
            try:
                cmd = q_cmd.get_nowait()
            except Empty:
                break
            if cmd == 'stop':
                should_stop = True
                q_cmd.put(True)
                break
            elif cmd == "state_dict":
                ret = OrderedDict()
                for c in config:
                    ds_name = "{}.{}".format(c["task_name"], c["dataset_name"])
                    ret[ds_name] = c["dataset"]._state_dict()
                q_cmd.put(ret)
            elif cmd == "load_state_dict":
                state_dict = q_cmd.get()
                missing = []
                for c in config:
                    ds_name = "{}.{}".format(c["task_name"], c["dataset_name"])
                    if ds_name in state_dict:
                        c["dataset"].load_state_dict(state_dict[ds_name])
                    else:
                        # new dataset
                        missing.append(ds_name)
                q_cmd.put(missing)
            elif cmd == "start":
                should_start = True
                q_cmd.put(True)
            else:
                raise RuntimeError("Unknown command: {}".format(cmd))
        
        if should_stop:
            break

        if not should_start:
            # wait for start cmd
            time.sleep(1)
            continue
            
        if len(config) == 0:
            # no dataset available
            time.sleep(1)
            continue

        # sample a dataset
        ds_id : int = 0

        while True:
            ds_id = np.random.choice(weights.shape[0], p=weights)
            if config[ds_id]["dataset"]._nlines != config[ds_id]["lines"]:
                # dataset size changed
                for c in config:
                    c["lines"] = c["dataset"]._nlines
                weights = _build_sample_weights(config)
                continue
            else:
                break
        
        batch = packer.add_data(config[ds_id])
        if batch is not None:
            # new batch comming
            q_data.put(batch)

class MixedDataset:
    def __init__(self,
            config_path : str,
            batch_size : int,
            max_length : int,
            tokenizer,
            incontext_sample_weight : Union[List[float], np.ndarray],
            max_depth : int = 16
        ) -> None:
        self._q_cmd = multiprocessing.Queue()
        self._q_data = multiprocessing.Queue()
        self._packer = _MixedDatasetBatchPacker(batch_size, max_length, tokenizer, incontext_sample_weight, max_depth)
        self._p = multiprocessing.Process(
            target=_mixed_dataset_process,
            args=(config_path, self._q_cmd, self._q_data, bmt.rank(), bmt.world_size(), self._packer),
        )
        self._p.start()
        self._closed = False
    
    def close(self):
        if not self._closed:
            self._closed = True
            self._q_cmd.put("stop")
            assert self._q_cmd.get(), "Failed to stop process"
            self._p.join()
    
    @property
    def closed(self):
        return self._closed
    
    def start(self):
        self._q_cmd.put("start")
        self._q_cmd.get()
    
    def state_dict(self):
        self._q_cmd.put("state_dict")
        states = self._q_cmd.get()
        if not isinstance(states, OrderedDict):
            raise RuntimeError("Invalid state dict {}".format(states))
        if bmt.world_size() == 1:
            return states

        ret = OrderedDict()
        for k, v in states.items():
            num_unused_block = v["states"].size(0)
            gpu_num_unused_block = torch.tensor([num_unused_block], dtype=torch.long).cuda()
            max_unused_blocks = bmt.distributed.all_reduce(gpu_num_unused_block, op="max").cpu().item()
            gpu_states = torch.full((max_unused_blocks,), -1, dtype=torch.long).cuda()
            gpu_states[:num_unused_block] = v["states"].cuda()

            gpu_block = v["block"].cuda()
            global_states = bmt.distributed.all_gather(gpu_states).cpu()    # (world_size, max_unused_blocks)
            global_block = bmt.distributed.all_gather(gpu_block).cpu()      # (world_size, 3)
            ret[k] = {"states": global_states, "block": global_block}
        return ret
    
    def load_state_dict(self, data : OrderedDict, strict : bool = False):
        self._q_cmd.put("load_state_dict")
        self._q_cmd.put(data)
        missing = self._q_cmd.get()
        if strict:
            if len(missing) > 0:
                raise RuntimeError("Missing dataset state: {}".format(missing))
        return missing

    def get(self):
        return self._q_data.get()
