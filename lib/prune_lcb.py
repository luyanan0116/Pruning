def prune_lcb(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    print("loading calibration data for LCB (Real Sequence Mode)")
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete")
    
    with torch.no_grad():
        inps, outs, attention_mask, position_ids, position_embeddings = prepare_calibration_input(model, dataloader, device)

    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if hasattr(model, "hf_device_map") and f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs = inps.to(dev), outs.to(dev)
            if attention_mask is not None: attention_mask = attention_mask.to(dev)
            if position_ids is not None: position_ids = position_ids.to(dev)
            if position_embeddings is not None:
                position_embeddings = (position_embeddings[0].to(dev), position_embeddings[1].to(dev))
                
        layer_kwargs = {}
        if attention_mask is not None: layer_kwargs["attention_mask"] = attention_mask
        if position_ids is not None: layer_kwargs["position_ids"] = position_ids
        if position_embeddings is not None: layer_kwargs["position_embeddings"] = position_embeddings

        wrapped_layers = {}
        # 【新增】字典用于存储当前层每个模块的真实时序激活
        real_activations = {}
        
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])
            real_activations[name] = [] # 初始化空列表

        def add_batch(name):
            def tmp(_, inp, out):
                # 兼容旧逻辑
                wrapped_layers[name].add_batch(inp[0].data, out.data)
                # 【核心拦截】捕获真实的输入序列 inp[0].data
                # 关键：务必使用 .cpu() 转移到主内存，否则 128 条长序列会瞬间撑爆 GPU 显存！
                real_activations[name].append(inp[0].data.cpu())
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
            
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name} using LCB")
            W = subset[name].weight.data
            
            # 【提取真实时序数据】
            # 将收集到的 CPU list 拼接为一个大的张量 (batch_size, seq_len, in_features)
            # 并在计算时转移回当前的计算设备 (dev)
            X_real = torch.cat(real_activations[name], dim=0).to(dev)
            
            # 调用 LCB 评估器：传入真实的 X_real 而不再是随机模拟数据
            W_metric = compute_lcb_weight_metric(W, X_real, args)

            # 计算完毕，立即清理庞大的真实序列数据，释放显存和内存
            del X_real
            real_activations[name] = None
            torch.cuda.empty_cache()

            # ----- 以下为标准掩码生成与裁剪逻辑 -----
            W_mask = (torch.zeros_like(W_metric) == 1)
            if prune_n != 0:
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)
                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha
                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                else:
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), **layer_kwargs)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()